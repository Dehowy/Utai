cmake_minimum_required(VERSION 3.22)
project(UtaiVSTHost VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE path
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../juce ${CMAKE_CURRENT_BINARY_DIR}/juce)

# Our VST host application
juce_add_gui_app(UtaiVSTHost
    PRODUCT_NAME "Utai VST Host"
    COMPANY_NAME "Utai"
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/../../electron/build/icon.png"
    ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/../../electron/build/icon.png"
)

target_sources(UtaiVSTHost PRIVATE
    src/Main.cpp
)

target_compile_definitions(UtaiVSTHost PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGINHOST_VST3=1
    JUCE_PLUGINHOST_VST=0
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:UtaiVSTHost,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:UtaiVSTHost,JUCE_VERSION>"
)

target_link_libraries(UtaiVSTHost PRIVATE
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags
)
