"""Minimal i18n stub for RVC training scripts.

The full RVC GUI has a complete i18n system with locale files.
This stub just returns the input string as-is, which is fine for
our headless training pipeline.
"""


class I18nAuto:
    def __init__(self):
        pass

    def __call__(self, key):
        return key
