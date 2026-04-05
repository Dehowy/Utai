using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;
using OpenUtau.Core;
using OpenUtau.Core.Format;
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using Serilog;

namespace UtaiRenderer;

class Program
{
    static int Main(string[] args)
    {
        // Register additional encodings (shift_jis, etc.)
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Information()
            .WriteTo.File("utai-renderer.log")
            .CreateLogger();

        if (args.Length < 1)
        {
            PrintUsage();
            return 1;
        }

        var command = args[0].ToLower();

        switch (command)
        {
            case "render":
                return HandleRender(args);
            case "singers":
                return HandleListSingers();
            case "info":
                return HandleInfo(args);
            default:
                Console.Error.WriteLine($"Unknown command: {command}");
                PrintUsage();
                return 1;
        }
    }

    static void PrintUsage()
    {
        Console.WriteLine("Utai Renderer - OpenUtau DiffSinger CLI");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  utai-renderer render <input.ustx> <output.wav> [--track <n>]");
        Console.WriteLine("  utai-renderer singers");
        Console.WriteLine("  utai-renderer info <input.ustx>");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  render   Render a .ustx project to WAV");
        Console.WriteLine("  singers  List available singers");
        Console.WriteLine("  info     Show project information");
    }

    static void InitializeCore()
    {
        DocManager.Inst.Initialize(Thread.CurrentThread, TaskScheduler.Default);
        // In console app, silently ignore UI thread notifications (progress bars etc.)
        // The render thread tries to post ProgressBarNotifications back to the UI thread,
        // which we don't have. Just drop them.
        DocManager.Inst.PostOnUIThread = action => { /* no-op in headless mode */ };
        SingerManager.Inst.Initialize();
    }

    static int HandleRender(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("Usage: utai-renderer render <input.ustx> <output.wav> [--track <n>]");
            return 1;
        }

        var inputPath = Path.GetFullPath(args[1]);
        var outputPath = Path.GetFullPath(args[2]);
        int trackNo = -1;

        for (int i = 3; i < args.Length; i++)
        {
            if (args[i] == "--track" && i + 1 < args.Length)
            {
                if (!int.TryParse(args[i + 1], out trackNo))
                {
                    Console.Error.WriteLine($"Invalid track number: {args[i + 1]}");
                    return 1;
                }
                i++;
            }
        }

        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input file not found: {inputPath}");
            return 1;
        }

        try
        {
            Console.WriteLine("Initializing...");
            InitializeCore();

            Console.WriteLine($"Loading project: {inputPath}");
            var project = Ustx.Load(inputPath);
            // Register project with DocManager so phonemizer runner can find it
            DocManager.Inst.ExecuteCmd(new LoadProjectNotification(project));

            Console.WriteLine($"Project: {project.name}");
            Console.WriteLine($"Tracks: {project.tracks.Count}");
            if (project.tempos.Count > 0)
                Console.WriteLine($"Tempo: {project.tempos[0].bpm} BPM");
            for (int i = 0; i < project.tracks.Count; i++)
            {
                var track = project.tracks[i];
                Console.WriteLine($"  Track {i}: {track.Singer?.Name ?? "(no singer)"} ({track.Singer?.SingerType})");
            }

            // Run phonemizer: ValidateFull pushes to PhonemizerRunner (async).
            // We poll until phonemes appear or timeout.
            Console.WriteLine("Phonemizing...");
            project.ValidateFull();
            int maxWait = 30; // seconds
            for (int wait = 0; wait < maxWait * 10; wait++)
            {
                DocManager.Inst.PhonemizerRunner?.WaitFinish();
                Thread.Sleep(100);
                // Check if any voice part now has phrases
                project.Validate(new OpenUtau.Core.ValidateOptions { SkipPhonemizer = true });
                bool hasPhrases = project.parts
                    .OfType<OpenUtau.Core.Ustx.UVoicePart>()
                    .Any(vp4 => vp4.GetRenderRequest()?.phrases?.Length > 0);
                if (hasPhrases)
                {
                    Console.WriteLine($"  Phonemization complete ({(wait + 1) * 100}ms)");
                    break;
                }
                if (wait % 10 == 0 && wait > 0) Console.Write(".");
            }

            var cancellation = new CancellationTokenSource();

            Console.WriteLine(trackNo >= 0
                ? $"Rendering track {trackNo}..."
                : "Rendering all tracks (mixdown)...");

            var engine = trackNo >= 0
                ? new RenderEngine(project, trackNo: trackNo)
                : new RenderEngine(project);

            Tuple<WaveMix, System.Collections.Generic.List<OpenUtau.Core.SignalChain.Fader>> result;
            try
            {
                result = engine.RenderMixdown(
                    TaskScheduler.Default,
                    ref cancellation,
                    wait: true);
            }
            catch (AggregateException ae)
            {
                foreach (var inner in ae.Flatten().InnerExceptions)
                {
                    Console.Error.WriteLine($"Render inner error: {inner.Message}");
                    Console.Error.WriteLine($"  at: {inner.StackTrace?.Split('\n').FirstOrDefault()?.Trim()}");
                }
                return 1;
            }

            var mixdown = result.Item1;


            var outputDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(outputDir))
                Directory.CreateDirectory(outputDir);

            WaveFileWriter.CreateWaveFile16(outputPath, new ExportAdapter(mixdown).ToMono(1, 0));

            Console.WriteLine($"OK: {outputPath}");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            Console.Error.WriteLine($"Stack: {ex.StackTrace}");
            if (ex.InnerException != null)
                Console.Error.WriteLine($"Inner: {ex.InnerException.Message}");
            Log.Error(ex, "Render failed");
            return 1;
        }
    }

    static int HandleListSingers()
    {
        try
        {
            InitializeCore();

            Console.WriteLine("Available singers:");
            Console.WriteLine();

            foreach (var kvp in SingerManager.Inst.Singers)
            {
                var singer = kvp.Value;
                Console.WriteLine($"  [{kvp.Key}] {singer.Name}");
                if (!string.IsNullOrEmpty(singer.Author))
                    Console.WriteLine($"    Author: {singer.Author}");
                Console.WriteLine($"    Type: {singer.SingerType}");
                Console.WriteLine();
            }

            if (SingerManager.Inst.Singers.Count == 0)
            {
                Console.WriteLine("  (no singers found)");
                Console.WriteLine($"  Singer search path: {PathManager.Inst.SingersPath}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    static int HandleInfo(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("Usage: utai-renderer info <input.ustx>");
            return 1;
        }

        var inputPath = Path.GetFullPath(args[1]);
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input file not found: {inputPath}");
            return 1;
        }

        try
        {
            InitializeCore();

            var project = Ustx.Load(inputPath);

            Console.WriteLine($"Project: {project.name}");
            Console.WriteLine($"File: {inputPath}");
            Console.WriteLine($"Tracks: {project.tracks.Count}");

            for (int i = 0; i < project.tracks.Count; i++)
            {
                var track = project.tracks[i];
                Console.WriteLine($"  Track {i}: {track.TrackName}");
                Console.WriteLine($"    Singer: {track.Singer?.Name ?? "(none)"}");
                var trackParts = project.parts.Count(p => p.trackNo == i);
                Console.WriteLine($"    Parts: {trackParts}");
            }

            if (project.tempos.Count > 0)
                Console.WriteLine($"Tempo: {project.tempos[0].bpm} BPM");
            if (project.timeSignatures.Count > 0)
                Console.WriteLine($"Time Signature: {project.timeSignatures[0].beatPerBar}/{project.timeSignatures[0].beatUnit}");

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }
}
