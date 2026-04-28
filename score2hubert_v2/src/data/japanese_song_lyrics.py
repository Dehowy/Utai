"""Hardcoded lyrics for 27 Japanese public-domain children's songs (童謡).

These are the songs used in JVS-MuSiC corpus:
- 1 common song: katatsumuri (sung by all 100 singers)
- 26 unique songs (each sung by 3-4 different singers)

All lyrics are public domain (copyright expired). Source: standard 童謡 lyrics
widely available online.

Hiragana form is used for forced alignment with vumichien/wav2vec2-large-xlsr-japanese-hiragana.
Punctuation/spaces are removed since the wav2vec2 model only outputs hiragana characters.
Multi-verse songs include all verses since singers typically sing the full song.
"""

# Map (song_name_in_jvs_singer_info) → hiragana lyrics (no spaces, no punctuation)
# Verse separators are kept implicit (concatenated). The model will align via CTC.

KATATSUMURI = (
    # Verse 1
    "でんでんむしむしかたつむり"
    "おまえのあたまはどこにある"
    "つのだせやりだせあたまだせ"
    # Verse 2
    "でんでんむしむしかたつむり"
    "おまえのめだまはどこにある"
    "つのだせやりだせめだまだせ"
)

UNIQUE_SONG_LYRICS = {
    # Most common (4 singers each)
    "あめふり": (
        "あめあめふれふれかあさんが"
        "じゃのめでおむかえうれしいな"
        "ぴっちぴっちちゃっぷちゃっぷらんらんらん"
    ),
    "うさぎとかめ": (
        "もしもしかめよかめさんよ"
        "せかいのうちでおまえほど"
        "あゆみののろいものはない"
        "どうしてそんなにのろいのか"
    ),
    "かごめかごめ": (
        "かごめかごめかごのなかのとりは"
        "いついつでやるよあけのばんに"
        "つるとかめがすべった"
        "うしろのしょうめんだあれ"
    ),
    "ずいずいずっころばし": (
        "ずいずいずっころばしごまみそずい"
        "ちゃつぼにおわれてとっぴんしゃん"
        "ぬけたらどんどこしょ"
        "たわらのねずみがこめくってちゅう"
        "ちゅうちゅうちゅう"
        "おっとさんがよんでもおっかさんがよんでも"
        "いきっこなしよ"
        "いどのまわりでおちゃわんかいたのだあれ"
    ),
    "ちょうちょう": (
        "ちょうちょうちょうちょうなのはにとまれ"
        "なのはにあいたらさくらにとまれ"
        "さくらのはなのさかゆくみよ"
        "さかゆくみよにとまれよあそべ"
    ),
    "はとぽっぽ": (
        "ぽっぽっぽはとぽっぽ"
        "まめがほしいかそらやるぞ"
        "みんなでなかよくたべにこい"
    ),
    "はるがきた": (
        "はるがきたはるがきたどこにきた"
        "やまにきたさとにきたのにもきた"
        "はながさくはながさくどこにさく"
        "やまにさくさとにさくのにもさく"
    ),
    "ふるさと": (
        "うさぎおいしかのやま"
        "こぶなつりしかのかわ"
        "ゆめはいまもめぐりて"
        "わすれがたきふるさと"
    ),
    "むすんでひらいて": (
        "むすんでひらいててをうってむすんで"
        "またひらいててをうってそのてをうえに"
        "むすんでひらいててをうってむすんで"
    ),
    "ゆき": (
        "ゆきやこんこあられやこんこ"
        "ふってはふってはずんずんつもる"
        "やまものはらもわたぼうしかぶり"
        "かれきのこらずはながさく"
    ),
    "われは海の子": (
        "われはうみのこしらなみの"
        "さわぐいそべのまつばらに"
        "けむりたなびくとまやこそ"
        "わがなつかしきすみかなれ"
    ),
    "ピクニック": (
        "おかをこえいこうよくちぶえふきつつ"
        "そらにはくものぼうけん"
        "うたごえたかくたびゆけば"
        "こだまはこたえんわれらのうたを"
    ),
    "一週間": (
        "にちようびにいちばにでかけ"
        "いとおとつむぎをかってきた"
        "つむぎつむぎつむぎつむぎつむぎいとおとつむぎ"
    ),
    "七つの子": (
        "からすなぜなくのからすはやまに"
        "かわいいななつのこがあるからよ"
        "かわいいかわいいとからすはなくの"
        "かわいいかわいいとなくんだよ"
    ),
    "子守り歌": (
        "ねんねんころりよおころりよ"
        "ぼうやはよいこだねんねしな"
        "ぼうやのおもりはどこへいた"
        "あのやまこえてさとへいた"
    ),
    "故郷の空": (
        "ゆうぞらはれてあきかぜふき"
        "つきかげおちてすずむしなく"
        "おもえばとおしこきょうのそら"
        "ああわがちちははいかにおわす"
    ),
    "早春賦": (
        "はるはなのみのかぜのさむさや"
        "たにのうぐいすうたはおもえど"
        "ときにあらずとこえもたてず"
        "ときにあらずとこえもたてず"
    ),
    "春の小川": (
        "はるのおがわはさらさらいくよ"
        "きしのすみれやれんげのはなに"
        "すがたやさしくいろうつくしく"
        "さけよさけよとささやきながら"
    ),
    "桃太郎": (
        "ももたろうさんももたろうさん"
        "おこしにつけたきびだんご"
        "ひとつわたしにくださいな"
    ),
    "茶摘み": (
        "なつもちかづくはちじゅうはちや"
        "のにもやまにもわかばがしげる"
        "あれにみえるはちゃつみじゃないか"
        "あかねだすきにすげのかさ"
    ),
    "虫のこえ": (
        "あれまつむしがないている"
        "ちんちろちんちろちんちろりん"
        "あれすずむしもなきだした"
        "りんりんりんりんりいんりん"
        "あきのよながをなきとおす"
        "ああおもしろいむしのこえ"
    ),
    "赤い靴": (
        "あかいくつはいてたおんなのこ"
        "いじんさんにつれられていっちゃった"
        "よこはまのはとばからふねにのって"
        "いじんさんにつれられていっちゃった"
    ),
    # Less common (3 singers each)
    "うさぎ": (
        "うさぎうさぎなにみてはねる"
        "じゅうごやおつきさまみてはねる"
    ),
    "げんこつやまのたぬきさん": (
        "げんこつやまのたぬきさん"
        "おっぱいのんでねんねして"
        "だっこしておんぶしてまたあした"
    ),
    "さくらさくら": (
        "さくらさくらやよいのそらは"
        "みわたすかぎりかすみかくもか"
        "においぞいずるいざやいざやみにゆかん"
    ),
    "蛍の光": (
        "ほたるのひかりまどのゆき"
        "ふみよむつきひかさねつつ"
        "いつしかとしもすぎのとを"
        "あけてぞけさはわかれゆく"
    ),
}


def get_lyrics(song_name: str) -> str:
    """Get hiragana lyrics for a JVS-MuSiC song name. Returns empty string if not found."""
    if song_name in UNIQUE_SONG_LYRICS:
        return UNIQUE_SONG_LYRICS[song_name]
    return ""


def get_katatsumuri() -> str:
    """Common song lyrics for all 100 JVS-MuSiC singers."""
    return KATATSUMURI


# Sanity check: print all lyrics if run directly
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(f"Katatsumuri ({len(KATATSUMURI)} chars):")
    print(f"  {KATATSUMURI}")
    print(f"\n{len(UNIQUE_SONG_LYRICS)} unique songs:")
    for name, lyrics in UNIQUE_SONG_LYRICS.items():
        print(f"  {name} ({len(lyrics)} chars): {lyrics[:50]}...")
