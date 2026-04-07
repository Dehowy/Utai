/*
    Utai VST Host — minimal standalone VST2/VST3 plugin host.

    Usage:  UtaiVSTHost.exe <plugin_path> [state_path] [owner_hwnd]

    - Opens the plugin's editor GUI with real-time audio output.
    - Blocks until the user closes the editor window.
    - Saves plugin state to state_path on close.
    - If owner_hwnd is provided, the editor window stays on top of it.
*/

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>

#ifdef _WIN32
 #include <windows.h>
#endif

//==============================================================================
class PluginWindow : public juce::DocumentWindow
{
public:
    PluginWindow(juce::AudioPluginInstance* plugin, juce::Component* editor,
                 juce::uint64 ownerHwnd, const juce::File& iconFile)
        : DocumentWindow("Utai \xe2\x80\x94 VST Editor",
                         juce::Colour(0xff0a0a1a),
                         DocumentWindow::closeButton | DocumentWindow::minimiseButton)
    {
        setUsingNativeTitleBar(true);
        setContentOwned(editor, true);
        setResizable(false, false);
        centreWithSize(editor->getWidth(), editor->getHeight());
        setVisible(true);
        toFront(true);

#ifdef _WIN32
        auto* peer = getPeer();
        if (peer != nullptr)
        {
            auto hwnd = (HWND) peer->getNativeHandle();

            // Set owner window (stays on top of main app)
            if (ownerHwnd != 0)
                ::SetWindowLongPtr(hwnd, GWLP_HWNDPARENT, (LONG_PTR) ownerHwnd);

            // Set Utai icon
            if (iconFile.existsAsFile())
            {
                HICON hIcon = (HICON) ::LoadImageW(
                    NULL, iconFile.getFullPathName().toWideCharPointer(),
                    IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE);
                if (hIcon)
                {
                    ::SendMessageW(hwnd, WM_SETICON, ICON_SMALL, (LPARAM) hIcon);
                    ::SendMessageW(hwnd, WM_SETICON, ICON_BIG, (LPARAM) hIcon);
                }
            }
        }
#endif
    }

    void closeButtonPressed() override
    {
        juce::JUCEApplication::getInstance()->systemRequestedQuit();
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginWindow)
};

//==============================================================================
class UtaiVSTHostApp : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override    { return "Utai VST Host"; }
    const juce::String getApplicationVersion() override { return "1.0.0"; }

    void initialise(const juce::String&) override
    {
        auto args = getCommandLineParameterArray();
        if (args.size() < 1)
        {
            std::cout << R"({"error":"Usage: UtaiVSTHost.exe <plugin_path> [state_path] [owner_hwnd]"})" << std::endl;
            quit();
            return;
        }

        pluginPath = args[0];
        statePath = args.size() > 1 ? args[1] : "";
        ownerHwnd = args.size() > 2 ? (juce::uint64) args[2].getLargeIntValue() : 0;

        // Resolve icon path (relative to exe → ../../electron/build/icon.ico)
        iconFile = juce::File::getSpecialLocation(juce::File::currentExecutableFile)
                       .getParentDirectory().getParentDirectory()
                       .getChildFile("electron").getChildFile("build").getChildFile("icon.ico");

        // Audio device
        juce::String err = deviceManager.initialiseWithDefaultDevices(0, 2);
        if (err.isNotEmpty())
        {
            std::cout << R"({"error":"Audio: )" << err.toStdString() << R"("})" << std::endl;
            quit();
            return;
        }

        // Plugin formats
        formatManager.addDefaultFormats();

        // Scan plugin file
        juce::OwnedArray<juce::PluginDescription> descs;
        juce::KnownPluginList kpl;
        for (auto* fmt : formatManager.getFormats())
            kpl.scanAndAddFile(pluginPath, true, descs, *fmt);

        if (descs.isEmpty())
        {
            std::cout << R"({"error":"Plugin not found: )" << pluginPath.toStdString() << R"("})" << std::endl;
            quit();
            return;
        }

        // Create instance
        juce::String loadErr;
        auto setup = deviceManager.getAudioDeviceSetup();
        double sr = setup.sampleRate > 0 ? setup.sampleRate : 44100.0;
        int bs = setup.bufferSize > 0 ? setup.bufferSize : 512;

        plugin = formatManager.createPluginInstance(*descs[0], sr, bs, loadErr);
        if (plugin == nullptr)
        {
            std::cout << R"({"error":"Load failed: )" << loadErr.toStdString() << R"("})" << std::endl;
            quit();
            return;
        }

        // Load saved state
        if (statePath.isNotEmpty())
        {
            juce::File sf(statePath);
            if (sf.existsAsFile())
            {
                juce::MemoryBlock mb;
                if (sf.loadFileAsData(mb))
                    plugin->setStateInformation(mb.getData(), (int) mb.getSize());
            }
        }

        // Connect audio
        player.setProcessor(plugin.get());
        deviceManager.addAudioCallback(&player);

        // Open editor
        if (plugin->hasEditor())
        {
            auto* ed = plugin->createEditor();
            if (ed != nullptr)
            {
                window = std::make_unique<PluginWindow>(plugin.get(), ed, ownerHwnd, iconFile);
                std::cout << R"({"status":"opened","name":")" << plugin->getName().toStdString()
                          << R"(","audio":true})" << std::endl;
                return;
            }
        }

        std::cout << R"({"error":"No editor GUI"})" << std::endl;
        quit();
    }

    void shutdown() override
    {
        // Save state
        if (plugin != nullptr && statePath.isNotEmpty())
        {
            juce::MemoryBlock mb;
            plugin->getStateInformation(mb);
            if (mb.getSize() > 0)
            {
                juce::File sf(statePath);
                sf.getParentDirectory().createDirectory();
                bool ok = sf.replaceWithData(mb.getData(), mb.getSize());
                std::cout << R"({"status":"closed","state_saved":)" << (ok ? "true" : "false") << "}" << std::endl;
            }
        }

        window.reset();
        deviceManager.removeAudioCallback(&player);
        player.setProcessor(nullptr);
        plugin.reset();
    }

    void systemRequestedQuit() override { quit(); }

private:
    juce::String pluginPath, statePath;
    juce::uint64 ownerHwnd = 0;
    juce::File iconFile;
    juce::AudioPluginFormatManager formatManager;
    juce::AudioDeviceManager deviceManager;
    juce::AudioProcessorPlayer player;
    std::unique_ptr<juce::AudioPluginInstance> plugin;
    std::unique_ptr<PluginWindow> window;
};

START_JUCE_APPLICATION(UtaiVSTHostApp)
