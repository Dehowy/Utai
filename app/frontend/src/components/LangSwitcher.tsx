import { useTranslation } from 'react-i18next'

const LANGS = [
  { code: 'zh', label: '中' },
  { code: 'ja', label: 'JA' },
  { code: 'en', label: 'EN' },
]

export function LangSwitcher() {
  const { i18n } = useTranslation()

  return (
    <div className="utai-lang-switcher">
      {LANGS.map((lang) => (
        <button
          key={lang.code}
          className={`utai-lang-switcher__item ${i18n.language === lang.code ? 'utai-lang-switcher__item--active' : ''}`}
          onClick={() => { i18n.changeLanguage(lang.code); localStorage.setItem('utai-lang', lang.code); }}
        >
          {lang.label}
        </button>
      ))}
    </div>
  )
}
