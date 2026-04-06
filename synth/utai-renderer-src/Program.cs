using System;
using System.Collections.Concurrent;
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
    // Simulates a UI message pump: executes callbacks synchronously but prevents
    // re-entrant stack overflow by queueing nested calls and draining iteratively.
    static readonly ConcurrentQueue<Action> _uiQueue = new();
    static bool _drainingUI = false;

    /// <summary>Enqueue a callback and drain if not already draining.</summary>
    static void EnqueueAndDrain(Action action)
    {
        _uiQueue.Enqueue(action);
        if (!_drainingUI)
        {
            _drainingUI = true;
            try
            {
                while (_uiQueue.TryDequeue(out var a))
                {
                    try { a(); } catch (Exception ex)
                    {
                        // Log but don't crash — UI notifications may fail in headless mode
                        Console.Error.WriteLine($"[warn] UI callback error: {ex.Message}");
                    }
                }
            }
            finally { _drainingUI = false; }
        }
    }

    /// <summary>Drain any remaining queued UI callbacks.</summary>
    static void DrainUIQueue()
    {
        // Force drain even if not currently draining (for explicit poll points)
        while (_uiQueue.TryDequeue(out var action))
        {
            try { action(); } catch { }
        }
    }

    static int Main(string[] args)
    {
        // Register additional encodings (shift_jis, etc.)
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

        // Write log next to the executable, not in cwd; rotate at 10 MB
        var exeDir = AppContext.BaseDirectory;
        var logPath = Path.Combine(exeDir, "utai-renderer.log");
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.File(logPath,
                fileSizeLimitBytes: 10 * 1024 * 1024,
                rollOnFileSizeLimit: true,
                retainedFileCountLimit: 2)
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

    /// <summary>Custom TaskScheduler that queues tasks for explicit drain on main thread.</summary>
    sealed class HeadlessTaskScheduler : TaskScheduler
    {
        public readonly ConcurrentQueue<Task> PendingTasks = new();
        protected override IEnumerable<Task> GetScheduledTasks() => PendingTasks;
        protected override void QueueTask(Task task) => PendingTasks.Enqueue(task);
        protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) => TryExecuteTask(task);
        public void DrainTasks()
        {
            while (PendingTasks.TryDequeue(out var task))
                TryExecuteTask(task);
        }
    }

    static readonly HeadlessTaskScheduler _headlessScheduler = new();

    static void InitializeCore()
    {
        // Enable GPU (DirectML) for ONNX inference if available
        OpenUtau.Core.Util.Preferences.Default.OnnxRunner = "DirectML";
        OpenUtau.Core.Util.Preferences.Default.DiffSingerTensorCache = true;
        try {
            var gpus = Onnx.getGpuInfo();
            // Pick the most capable GPU (prefer discrete over integrated)
            int bestGpu = 0;
            for (int i = 0; i < gpus.Count; i++) {
                Console.WriteLine($"  GPU[{i}]: {gpus[i]}");
                var desc = gpus[i].description.ToLower();
                if (desc.Contains("nvidia") || desc.Contains("geforce") || desc.Contains("rtx") || desc.Contains("gtx")) {
                    bestGpu = i;
                }
            }
            OpenUtau.Core.Util.Preferences.Default.OnnxGpu = bestGpu;
            Console.WriteLine($"Using GPU[{bestGpu}]: {(bestGpu < gpus.Count ? gpus[bestGpu].ToString() : "default")}");
        } catch (Exception ex) {
            Console.WriteLine($"GPU detection failed ({ex.Message}), CPU fallback");
            OpenUtau.Core.Util.Preferences.Default.OnnxRunner = "CPU";
        }

        // Use custom scheduler so PhonemizerRunner.SendResponse tasks land
        // in a queue we can drain from the main thread (not thread pool).
        DocManager.Inst.Initialize(Thread.CurrentThread, _headlessScheduler);
        DocManager.Inst.PostOnUIThread = action => EnqueueAndDrain(action);
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
            DrainUIQueue();

            Console.WriteLine($"Project: {project.name}");
            Console.WriteLine($"Tracks: {project.tracks.Count}");
            if (project.tempos.Count > 0)
                Console.WriteLine($"Tempo: {project.tempos[0].bpm} BPM");
            for (int i = 0; i < project.tracks.Count; i++)
            {
                var track = project.tracks[i];
                Console.WriteLine($"  Track {i}: {track.Singer?.Name ?? "(no singer)"} ({track.Singer?.SingerType})");
            }

            // Pre-flight: check that required DiffSinger model files exist
            foreach (var vp in project.parts.OfType<OpenUtau.Core.Ustx.UVoicePart>())
            {
                var singer = project.tracks[vp.trackNo].Singer;
                if (singer == null) continue;
                var singerDir = singer.Location;
                if (!string.IsNullOrEmpty(singerDir))
                {
                    // Check files/ directory at singer root, parent, or sibling
                    bool hasFiles = Directory.Exists(Path.Combine(singerDir, "files"))
                        || Directory.Exists(Path.Combine(singerDir, "..", "files"))
                        || Directory.Exists(Path.Combine(Path.GetDirectoryName(singerDir) ?? singerDir, "files"));
                    if (!hasFiles)
                    {
                        Console.Error.WriteLine($"Error: Voice bank model files not found.");
                        Console.Error.WriteLine($"  Expected: files/ directory near {singerDir}");
                        Console.Error.WriteLine($"  The DiffSinger voice bank '{singer.Name}' has config but no ONNX models.");
                        Console.Error.WriteLine($"  Please download the full voice bank (with .onnx models) via the Model Manager.");
                        return 1;
                    }
                }
            }

            // Run phonemizer via PhonemizerRunner (async) with custom TaskScheduler drain.
            // ValidateFull pushes requests; we poll + drain until phrases appear.
            Console.WriteLine("Phonemizing...");
            project.ValidateFull();
            DrainUIQueue();
            _headlessScheduler.DrainTasks();
            DrainUIQueue();

            int maxWait = 120;
            bool phonemized = false;
            for (int wait = 0; wait < maxWait * 10; wait++)
            {
                Thread.Sleep(100);
                _headlessScheduler.DrainTasks();
                DrainUIQueue();

                project.Validate(new OpenUtau.Core.ValidateOptions { SkipPhonemizer = true });
                bool hasPhrases = project.parts
                    .OfType<OpenUtau.Core.Ustx.UVoicePart>()
                    .Any(vp4 => vp4.GetRenderRequest()?.phrases?.Length > 0);
                if (hasPhrases)
                {
                    Console.WriteLine($"  Phonemization complete ({(wait + 1) * 100}ms)");
                    phonemized = true;
                    break;
                }
                if (wait == 10)
                {
                    Console.WriteLine($"  [1s] scheduler pending={_headlessScheduler.PendingTasks.Count}, uiQ={_uiQueue.Count}");
                    // Dump note/phoneme state for debugging
                    foreach (var vp in project.parts.OfType<OpenUtau.Core.Ustx.UVoicePart>())
                    {
                        Console.WriteLine($"  Part '{vp.name}': {vp.notes.Count} notes, trackNo={vp.trackNo}");
                        if (vp.notes.Count > 0)
                        {
                            var first = vp.notes.First();
                            Console.WriteLine($"    First note: lyric='{first.lyric}' tone={first.tone} pos={first.position}");
                        }
                    }
                }
                if (wait == 50) Console.WriteLine($"  [5s] scheduler pending={_headlessScheduler.PendingTasks.Count}, uiQ={_uiQueue.Count}");
                if (wait % 100 == 0 && wait > 0) Console.Write(".");
            }

            if (!phonemized)
            {
                Console.Error.WriteLine("Error: Phonemization timed out after 120s. Diagnostics:");
                Console.Error.WriteLine($"  scheduler pending={_headlessScheduler.PendingTasks.Count}, uiQ={_uiQueue.Count}");
                // Check each singer's model files for the user
                foreach (var vp in project.parts.OfType<OpenUtau.Core.Ustx.UVoicePart>())
                {
                    var singer = project.tracks[vp.trackNo].Singer;
                    if (singer == null) { Console.Error.WriteLine($"  Track {vp.trackNo}: NO SINGER assigned"); continue; }
                    Console.Error.WriteLine($"  Track {vp.trackNo}: singer='{singer.Name}' location='{singer.Location}'");
                    var filesDir = Path.Combine(singer.Location, "files");
                    Console.Error.WriteLine($"    files/ exists: {Directory.Exists(filesDir)}");
                    if (Directory.Exists(filesDir))
                    {
                        var phonemesJson = Path.Combine(filesDir, "phonemes.json");
                        var languagesJson = Path.Combine(filesDir, "languages.json");
                        Console.Error.WriteLine($"    phonemes.json: {File.Exists(phonemesJson)}");
                        Console.Error.WriteLine($"    languages.json: {File.Exists(languagesJson)}");
                    }
                }
                Console.Error.WriteLine("  Possible causes:");
                Console.Error.WriteLine("    - Voice bank ONNX model files missing (phonemes.json, languages.json, *.onnx)");
                Console.Error.WriteLine("    - Incompatible phonemizer for the voice bank");
                Console.Error.WriteLine("    - Path resolution error in dsconfig.yaml (check relative paths)");
                Console.Error.WriteLine($"  Check log file for details: {Path.Combine(AppContext.BaseDirectory, "utai-renderer.log")}");
                return 1;
            }

            var cancellation = new CancellationTokenSource();

            Console.WriteLine(trackNo >= 0
                ? $"Rendering track {trackNo}..."
                : "Rendering all tracks (mixdown)...");

            var engine = trackNo >= 0
                ? new RenderEngine(project, trackNo: trackNo)
                : new RenderEngine(project);

            // Run the blocking RenderMixdown on a background thread so we can
            // drain UI + scheduler callbacks on the main thread (avoids deadlock).
            Tuple<WaveMix, System.Collections.Generic.List<OpenUtau.Core.SignalChain.Fader>> result = null;
            Exception renderError = null;

            var renderThread = new Thread(() =>
            {
                try
                {
                    result = engine.RenderMixdown(
                        _headlessScheduler,
                        ref cancellation,
                        wait: true);
                }
                catch (Exception ex)
                {
                    renderError = ex;
                }
            });
            renderThread.IsBackground = true;
            renderThread.Start();

            // Main thread: drain callbacks while render runs
            int renderMaxWait = 300 * 10; // 300 seconds
            for (int rw = 0; rw < renderMaxWait && renderThread.IsAlive; rw++)
            {
                Thread.Sleep(100);
                _headlessScheduler.DrainTasks();
                DrainUIQueue();
                if (rw % 100 == 0 && rw > 0) Console.Write(".");
            }

            if (renderThread.IsAlive)
            {
                cancellation.Cancel();
                renderThread.Join(5000);
                Console.Error.WriteLine("\nError: Rendering timed out (300s)");
                return 1;
            }

            if (renderError != null)
            {
                var ae = renderError as AggregateException;
                if (ae != null)
                {
                    foreach (var inner in ae.Flatten().InnerExceptions)
                    {
                        Console.Error.WriteLine($"Render error: {inner.Message}");
                        Console.Error.WriteLine($"  at: {inner.StackTrace?.Split('\n').FirstOrDefault()?.Trim()}");
                    }
                }
                else
                {
                    Console.Error.WriteLine($"Render error: {renderError.Message}");
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
