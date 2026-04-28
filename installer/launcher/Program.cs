using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

var tempDir = Path.Combine(Path.GetTempPath(), "UtaiInstaller_" + DateTime.Now.Ticks.ToString().Substring(10));
Directory.CreateDirectory(tempDir);

try
{
    var asm = Assembly.GetExecutingAssembly();

    var szPath = Path.Combine(tempDir, "7za.exe");
    using (var s = asm.GetManifestResourceStream("7za.exe")!)
    using (var f = File.Create(szPath))
        s.CopyTo(f);

    var archivePath = Path.Combine(tempDir, "content.7z");
    using (var s = asm.GetManifestResourceStream("content.7z")!)
    using (var f = File.Create(archivePath))
        s.CopyTo(f);

    var extractDir = Path.Combine(tempDir, "app");
    Directory.CreateDirectory(extractDir);
    var psi = new ProcessStartInfo(szPath, $"x \"{archivePath}\" -o\"{extractDir}\" -y")
    {
        UseShellExecute = false,
        CreateNoWindow = true,
    };
    Process.Start(psi)?.WaitForExit();

    File.Delete(archivePath);

    var installerExe = Path.Combine(extractDir, "Utai Installer.exe");
    if (File.Exists(installerExe))
    {
        var p = Process.Start(installerExe);
        p?.WaitForExit();
    }
}
finally
{
    try { Directory.Delete(tempDir, true); } catch { }
}
