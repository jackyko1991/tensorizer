/** Easter egg: historical scientists named in problem text get a dotted underline and a pixel-portrait card.
 *  Only people who have died; the one-liners are jokes in their voice, not quotations. Portraits are 16×16 sprites
 *  assembled from a few features (hair, beard, glasses, prop) in the site's muted palette. */

export interface Scientist {
  id: string;
  name: string;
  years: string;
  /** Spellings to find in the text, in any language (longest match wins). */
  aliases: string[];
  /** Joke one-liner by language; missing languages fall back to English. */
  meme: Partial<Record<string, string>> & { en: string };
  look: Look;
}

interface Look {
  hair: "short" | "wild" | "bald" | "side" | "long" | "receding" | "curly";
  hairColor: string;
  beard?: "full" | "mustache" | "goatee" | "sideburns";
  beardColor?: string;
  glasses?: "round" | "square";
  coat: string;
  prop?: "box" | "apple" | "cube" | "tape" | "flask";
}

const WHITE = "#e5e7eb";
const GREY = "#9ca3af";
const DARK = "#374151";
const BROWN = "#6b4f3a";
const NAVY = "#1e3a8a";
const SLATE = "#475569";
const TEAL = "#3f6f73";
const PLUM = "#6b4c6e";

export const SCIENTISTS: Scientist[] = [
  {
    id: "ramanujan",
    name: "Srinivasa Ramanujan",
    years: "1887–1920",
    aliases: ["Ramanujan", "拉馬努金", "拉马努金", "ラマヌジャン", "Рамануджан", "라마누잔", "रामानुजन"],
    meme: {
      en: 'Hardy\'s taxi was number 1729, "rather dull"? It is the smallest number that is a sum of two cubes in two ways.',
      "zh-Hant": "哈代說計程車號碼 1729 很無聊？它是能用兩種方式寫成兩個立方數之和的最小正整數。",
      "zh-Hans": "哈代说出租车号码 1729 很无聊？它是能用两种方式写成两个立方数之和的最小正整数。",
    },
    look: { hair: "side", hairColor: DARK, coat: NAVY, prop: "cube" },
  },
  {
    id: "markov",
    name: "Andrey Markov",
    years: "1856–1922",
    aliases: ["Markov", "馬可夫", "马尔可夫", "馬爾可夫", "马可夫", "マルコフ", "Марков", "마르코프", "मार्कोव"],
    meme: {
      en: "My future depends only on the present. What was I saying before? No idea. Memoryless.",
      "zh-Hant": "我的未來只取決於現在，跟過去毫無關係。剛才說到哪？忘了。無記憶性。",
      "zh-Hans": "我的未来只取决于现在，跟过去毫无关系。刚才说到哪？忘了。无记忆性。",
    },
    look: { hair: "receding", hairColor: GREY, beard: "full", beardColor: WHITE, glasses: "square", coat: SLATE },
  },
  {
    id: "schrodinger",
    name: "Erwin Schrödinger",
    years: "1887–1961",
    aliases: [
      "Schrödinger",
      "Schrodinger",
      "薛丁格",
      "薛定谔",
      "薛定諤",
      "シュレーディンガー",
      "Шрёдингер",
      "슈뢰딩거",
      "श्रोडिंगर",
    ],
    meme: {
      en: "Don't open this card: until you do, the cat in the box is both alive and dead.",
      "zh-Hant": "不要打開這張卡片，不然盒子裡的貓就不再是又死又活了。",
      "zh-Hans": "不要打开这张卡片，不然盒子里的猫就不再是又死又活了。",
    },
    look: { hair: "short", hairColor: DARK, glasses: "round", coat: TEAL, prop: "box" },
  },
  {
    id: "einstein",
    name: "Albert Einstein",
    years: "1879–1955",
    aliases: ["Einstein", "愛因斯坦", "爱因斯坦", "アインシュタイン", "Эйнштейн", "아인슈타인", "आइंस्टीन"],
    meme: {
      en: "Gravity isn't a pull: the spacetime under your feet is curved and you are just falling along it.",
      "zh-Hant": "重力不是拉力，是你腳下的時空彎了，你只是順著它往下掉。",
      "zh-Hans": "引力不是拉力，是你脚下的时空弯了，你只是顺着它往下掉。",
    },
    look: { hair: "wild", hairColor: WHITE, beard: "mustache", beardColor: WHITE, coat: SLATE },
  },
  {
    id: "newton",
    name: "Isaac Newton",
    years: "1643–1727",
    aliases: ["Newton", "牛頓", "牛顿", "ニュートン", "Ньютон", "뉴턴", "न्यूटन"],
    meme: {
      en: "For every action there is an equal and opposite reaction. Including from the apple.",
      "zh-Hant": "每個作用力都有大小相等、方向相反的反作用力，蘋果也不例外。",
      "zh-Hans": "每个作用力都有大小相等、方向相反的反作用力，苹果也不例外。",
    },
    look: { hair: "long", hairColor: GREY, coat: PLUM, prop: "apple" },
  },
  {
    id: "gauss",
    name: "Carl Friedrich Gauss",
    years: "1777–1855",
    aliases: ["Gauss", "高斯", "ガウス", "Гаусс", "가우스", "गॉस"],
    meme: {
      en: "1 + 2 + … + 100? 5050. Next question, please: I have a 17-gon to construct.",
      "zh-Hant": "1 加到 100？5050。下一題，我還要去作正十七邊形。",
      "zh-Hans": "1 加到 100？5050。下一题，我还要去作正十七边形。",
    },
    look: { hair: "short", hairColor: WHITE, coat: NAVY },
  },
  {
    id: "euler",
    name: "Leonhard Euler",
    years: "1707–1783",
    aliases: ["Euler", "歐拉", "欧拉", "オイラー", "Эйлер", "오일러", "ऑयलर"],
    meme: {
      en: "e^{iπ} + 1 = 0. Five constants, one line, no extra charge.",
      "zh-Hant": "e^{iπ} + 1 = 0：五個常數，一行搞定，不另收費。",
      "zh-Hans": "e^{iπ} + 1 = 0：五个常数，一行搞定，不另收费。",
    },
    look: { hair: "long", hairColor: WHITE, coat: BROWN },
  },
  {
    id: "riemann",
    name: "Bernhard Riemann",
    years: "1826–1866",
    aliases: ["Riemann", "黎曼", "リーマン", "Риман", "리만", "रीमान"],
    meme: {
      en: "All the nontrivial zeros are on the line Re s = ½. Probably. Proof left as an exercise (since 1859).",
      "zh-Hant": "所有非平凡零點都在 Re s = ½ 上。大概吧。證明留作習題（自 1859 年起）。",
      "zh-Hans": "所有非平凡零点都在 Re s = ½ 上。大概吧。证明留作习题（自 1859 年起）。",
    },
    look: { hair: "short", hairColor: DARK, beard: "full", beardColor: DARK, coat: SLATE },
  },
  {
    id: "fourier",
    name: "Joseph Fourier",
    years: "1768–1830",
    aliases: ["Fourier", "傅立葉", "傅里叶", "傅里葉", "フーリエ", "Фурье", "푸리에", "फ़ूरियर"],
    meme: {
      en: "Any shape is just enough circles going round. Even you.",
      "zh-Hant": "任何形狀都只是夠多的圓在轉。你也是。",
      "zh-Hans": "任何形状都只是够多的圆在转。你也是。",
    },
    look: { hair: "curly", hairColor: BROWN, coat: NAVY },
  },
  {
    id: "maxwell",
    name: "James Clerk Maxwell",
    years: "1831–1879",
    aliases: ["Maxwell", "馬克士威", "麦克斯韦", "麥克斯韋", "マクスウェル", "Максвелл", "맥스웰", "मैक्सवेल"],
    meme: {
      en: "Light? Just electricity and magnetism chasing each other at 3 × 10⁸ m/s. Also, about my demon…",
      "zh-Hant": "光？不過是電和磁以每秒三億米互相追逐。還有，關於我的那隻小妖……",
      "zh-Hans": "光？不过是电和磁以每秒三亿米互相追逐。还有，关于我的那只小妖……",
    },
    look: { hair: "curly", hairColor: DARK, beard: "full", beardColor: DARK, coat: TEAL },
  },
  {
    id: "boltzmann",
    name: "Ludwig Boltzmann",
    years: "1844–1906",
    aliases: ["Boltzmann", "波茲曼", "玻尔兹曼", "玻爾茲曼", "ボルツマン", "Больцман", "볼츠만", "बोल्ट्ज़मान"],
    meme: {
      en: "S = k log W. It's engraved on my tombstone, so it must be true.",
      "zh-Hant": "S = k log W。都刻在我墓碑上了，總不會錯吧。",
      "zh-Hans": "S = k log W。都刻在我墓碑上了，总不会错吧。",
    },
    look: { hair: "receding", hairColor: BROWN, beard: "full", beardColor: BROWN, glasses: "round", coat: SLATE },
  },
  {
    id: "planck",
    name: "Max Planck",
    years: "1858–1947",
    aliases: ["Planck", "普朗克", "プランク", "Планк", "플랑크", "प्लांक"],
    meme: {
      en: "I only quantised the energy as an act of desperation. It worked a little too well.",
      "zh-Hant": "我把能量量子化只是走投無路的權宜之計，沒想到效果太好了。",
      "zh-Hans": "我把能量量子化只是走投无路的权宜之计，没想到效果太好了。",
    },
    look: { hair: "bald", hairColor: GREY, beard: "mustache", beardColor: GREY, glasses: "round", coat: DARK },
  },
  {
    id: "bohr",
    name: "Niels Bohr",
    years: "1885–1962",
    aliases: ["Bohr", "波耳", "玻尔", "玻爾", "ボーア", "Бор", "보어", "बोर"],
    meme: {
      en: "If quantum mechanics hasn't shocked you, you haven't understood it. Electrons: stay in your orbits.",
      "zh-Hant": "如果量子力學沒嚇到你，你就還沒懂。電子們，乖乖待在軌道上。",
      "zh-Hans": "如果量子力学没吓到你，你就还没懂。电子们，乖乖待在轨道上。",
    },
    look: { hair: "receding", hairColor: BROWN, coat: SLATE },
  },
  {
    id: "heisenberg",
    name: "Werner Heisenberg",
    years: "1901–1976",
    aliases: ["Heisenberg", "海森堡", "海森伯", "ハイゼンベルク", "Гейзенберг", "하이젠베르크", "हाइज़ेनबर्ग"],
    meme: {
      en: "I know exactly where this card is. So I have no idea how fast it's moving.",
      "zh-Hant": "我很確定這張卡片在哪裡，所以完全不知道它跑多快。",
      "zh-Hans": "我很确定这张卡片在哪里，所以完全不知道它跑多快。",
    },
    look: { hair: "side", hairColor: BROWN, coat: NAVY },
  },
  {
    id: "dirac",
    name: "Paul Dirac",
    years: "1902–1984",
    aliases: ["Dirac", "狄拉克", "ディラック", "Дирак", "디랙", "डिराक"],
    meme: {
      en: "My equation said there should be an anti-electron. I said nothing. (I rarely do.)",
      "zh-Hant": "我的方程說應該有反電子。我什麼都沒說。（我很少說話。）",
      "zh-Hans": "我的方程说应该有反电子。我什么都没说。（我很少说话。）",
    },
    look: { hair: "side", hairColor: DARK, coat: SLATE },
  },
  {
    id: "feynman",
    name: "Richard Feynman",
    years: "1918–1988",
    aliases: ["Feynman", "費曼", "费曼", "ファインマン", "Фейнман", "파인만", "फ़ाइनमैन"],
    meme: {
      en: "Sum over every possible path. Yes, including the one through the bongo drum.",
      "zh-Hant": "把所有可能的路徑都加起來。對，包括經過邦哥鼓的那一條。",
      "zh-Hans": "把所有可能的路径都加起来。对，包括经过邦戈鼓的那一条。",
    },
    look: { hair: "curly", hairColor: GREY, coat: TEAL },
  },
  {
    id: "hawking",
    name: "Stephen Hawking",
    years: "1942–2018",
    aliases: ["Hawking", "霍金", "ホーキング", "Хокинг", "호킹", "हॉकिंग"],
    meme: {
      en: "Black holes ain't so black. Give them time: they glow, and they evaporate.",
      "zh-Hant": "黑洞沒那麼黑。給它一點時間：它會發光，然後蒸發。",
      "zh-Hans": "黑洞没那么黑。给它一点时间：它会发光，然后蒸发。",
    },
    look: { hair: "side", hairColor: BROWN, glasses: "square", coat: NAVY },
  },
  {
    id: "turing",
    name: "Alan Turing",
    years: "1912–1954",
    aliases: ["Turing", "圖靈", "图灵", "チューリング", "Тьюринг", "튜링", "ट्यूरिंग"],
    meme: {
      en: "Will this program ever halt? There is no general way to know. Try it and see.",
      "zh-Hant": "這個程式會停嗎？沒有通用的方法能知道。跑跑看吧。",
      "zh-Hans": "这个程序会停吗？没有通用的方法能知道。跑跑看吧。",
    },
    look: { hair: "side", hairColor: DARK, coat: SLATE, prop: "tape" },
  },
  {
    id: "kepler",
    name: "Johannes Kepler",
    years: "1571–1630",
    aliases: ["Kepler", "克卜勒", "开普勒", "開普勒", "ケプラー", "Кеплер", "케플러", "केप्लर"],
    meme: {
      en: "Circles were beautiful. The data said ellipses. I went with the data.",
      "zh-Hant": "圓很美，但數據說是橢圓。我選擇相信數據。",
      "zh-Hans": "圆很美，但数据说是椭圆。我选择相信数据。",
    },
    look: { hair: "short", hairColor: BROWN, beard: "goatee", beardColor: BROWN, coat: PLUM },
  },
  {
    id: "galileo",
    name: "Galileo Galilei",
    years: "1564–1642",
    aliases: ["Galileo", "伽利略", "ガリレオ", "Галилей", "갈릴레이", "गैलीलियो"],
    meme: {
      en: "Drop a heavy ball and a light ball together. Same time. And yet it moves.",
      "zh-Hant": "重球和輕球一起放手，同時落地。而且，它確實在動。",
      "zh-Hans": "重球和轻球一起放手，同时落地。而且，它确实在动。",
    },
    look: { hair: "receding", hairColor: BROWN, beard: "full", beardColor: BROWN, coat: BROWN },
  },
  {
    id: "lagrange",
    name: "Joseph-Louis Lagrange",
    years: "1736–1813",
    aliases: ["Lagrange", "拉格朗日", "ラグランジュ", "Лагранж", "라그랑주", "लग्रांज"],
    meme: {
      en: "Forces? Too fiddly. Write down T − V and let the calculus of variations do the pushing.",
      "zh-Hant": "力？太麻煩。寫下 T − V，讓變分法替你推。",
      "zh-Hans": "力？太麻烦。写下 T − V，让变分法替你推。",
    },
    look: { hair: "long", hairColor: WHITE, coat: NAVY },
  },
  {
    id: "hamilton",
    name: "William Rowan Hamilton",
    years: "1805–1865",
    aliases: ["Hamilton", "哈密頓", "哈密顿", "ハミルトン", "Гамильтон", "해밀턴", "हैमिल्टन"],
    meme: {
      en: "i² = j² = k² = ijk = −1. I carved it into a bridge so I wouldn't forget.",
      "zh-Hant": "i² = j² = k² = ijk = −1。我把它刻在橋上，免得忘了。",
      "zh-Hans": "i² = j² = k² = ijk = −1。我把它刻在桥上，免得忘了。",
    },
    look: { hair: "curly", hairColor: BROWN, beard: "sideburns", beardColor: BROWN, coat: TEAL },
  },
  {
    id: "poincare",
    name: "Henri Poincaré",
    years: "1854–1912",
    aliases: ["Poincaré", "Poincare", "龐加萊", "庞加莱", "ポアンカレ", "Пуанкаре", "푸앵카레", "पॉइनकेयर"],
    meme: {
      en: "Three bodies? I won a king's prize showing nobody can solve that in closed form. You're welcome, chaos.",
      "zh-Hant": "三體？我證明沒人能寫出解析解，還因此拿了國王的獎。不客氣，混沌。",
      "zh-Hans": "三体？我证明没人能写出解析解，还因此拿了国王的奖。不客气，混沌。",
    },
    look: { hair: "short", hairColor: GREY, beard: "full", beardColor: GREY, glasses: "round", coat: SLATE },
  },
  {
    id: "hilbert",
    name: "David Hilbert",
    years: "1862–1943",
    aliases: ["Hilbert", "希爾伯特", "希尔伯特", "ヒルベルト", "Гильберт", "힐베르트", "हिल्बर्ट"],
    meme: {
      en: "Here are 23 problems for the next century. We must know, we will know. (Gödel: about that…)",
      "zh-Hant": "給下個世紀的 23 個問題。我們必須知道，我們必將知道。（哥德爾：這個嘛……）",
      "zh-Hans": "给下个世纪的 23 个问题。我们必须知道，我们必将知道。（哥德尔：这个嘛……）",
    },
    look: { hair: "bald", hairColor: GREY, beard: "goatee", beardColor: GREY, glasses: "round", coat: DARK },
  },
  {
    id: "fermat",
    name: "Pierre de Fermat",
    years: "1607–1665",
    aliases: ["Fermat", "費馬", "费马", "フェルマー", "Ферма", "페르마", "फ़र्मा"],
    meme: {
      en: "I have a truly marvellous proof, which this card is too small to contain.",
      "zh-Hant": "我有一個絕妙的證明，可惜這張卡片太小，寫不下。",
      "zh-Hans": "我有一个绝妙的证明，可惜这张卡片太小，写不下。",
    },
    look: { hair: "long", hairColor: DARK, beard: "mustache", beardColor: DARK, coat: PLUM },
  },
  {
    id: "noether",
    name: "Emmy Noether",
    years: "1882–1935",
    aliases: ["Noether", "諾特", "诺特", "ネーター", "Нётер", "뇌터", "नोएदर"],
    meme: {
      en: "Every symmetry hides a conservation law. Time doesn't care when you start, so energy is conserved.",
      "zh-Hant": "每個對稱都藏著一條守恆律。時間不在乎你何時開始，所以能量守恆。",
      "zh-Hans": "每个对称都藏着一条守恒律。时间不在乎你何时开始，所以能量守恒。",
    },
    look: { hair: "side", hairColor: DARK, glasses: "round", coat: TEAL },
  },
  {
    id: "shannon",
    name: "Claude Shannon",
    years: "1916–2001",
    aliases: ["Shannon", "夏農", "香农", "シャノン", "Шеннон", "섀넌", "शैनन"],
    meme: {
      en: "Information is surprise, measured in bits. This card: about 0.3 bits. Sorry.",
      "zh-Hant": "資訊就是驚訝，以位元計。這張卡片：大約 0.3 位元。抱歉。",
      "zh-Hans": "信息就是惊讶，以比特计。这张卡片：大约 0.3 比特。抱歉。",
    },
    look: { hair: "short", hairColor: GREY, coat: NAVY },
  },
];

/** One regex over every alias (longest first, so 玻尔兹曼 wins over 玻尔). Latin names need word edges. */
export const ALIAS_INDEX: Map<string, Scientist> = new Map(
  SCIENTISTS.flatMap((s) => s.aliases.map((a) => [a, s] as const)),
);
const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
export const ALIAS_RE = new RegExp(
  [...ALIAS_INDEX.keys()]
    .sort((a, b) => b.length - a.length)
    .map((a) => (/^[A-Za-zÀ-ÿ]/.test(a) ? `(?<![A-Za-zÀ-ÿ])${escapeRe(a)}(?![A-Za-zÀ-ÿ])` : escapeRe(a)))
    .join("|"),
  "g",
);

// ---------- 16×16 pixel portraits ----------

const SKIN = "#e8c4a0";
const SKIN_DARK = "#b98a62";
const CREAM = "#f5f1e6";
const INK = "#1f2937";

/** The portrait as a 16×16 grid of colours ("" = background). */
export function portrait(s: Scientist): string[][] {
  const g: string[][] = Array.from({ length: 16 }, () => Array(16).fill(""));
  const fill = (x0: number, x1: number, y0: number, y1: number, c: string) => {
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (x >= 0 && x < 16 && y >= 0 && y < 16) g[y][x] = c;
  };
  const px = (x: number, y: number, c: string) => x >= 0 && x < 16 && y >= 0 && y < 16 && (g[y][x] = c);
  const L = s.look;
  const skin = s.id === "ramanujan" ? SKIN_DARK : SKIN;

  // shoulders and collar
  fill(2, 13, 13, 15, L.coat);
  fill(6, 9, 13, 13, CREAM);
  px(7, 14, CREAM);
  px(8, 14, CREAM);
  // neck and face
  fill(6, 9, 11, 12, skin);
  fill(4, 11, 4, 10, skin);
  for (const [x, y] of [
    [4, 4],
    [11, 4],
    [4, 10],
    [11, 10],
  ])
    px(x, y, "");
  px(3, 7, skin); // ears
  px(12, 7, skin);
  // eyes and mouth
  px(6, 7, INK);
  px(9, 7, INK);
  fill(7, 8, 9, 9, "#9a5b4b");

  // hair
  const h = L.hairColor;
  if (L.hair === "short") (fill(4, 11, 3, 4, h), px(4, 5, h), px(11, 5, h));
  if (L.hair === "side") (fill(4, 11, 3, 4, h), fill(4, 6, 5, 5, h), px(11, 5, h));
  if (L.hair === "curly") {
    fill(4, 11, 2, 4, h);
    for (const x of [3, 5, 7, 9, 11, 12]) px(x, 2 + (x % 2), h);
    (px(3, 5, h), px(12, 5, h));
  }
  if (L.hair === "long")
    (fill(4, 11, 2, 3, h), fill(2, 3, 3, 12, h), fill(12, 13, 3, 12, h), px(4, 4, h), px(11, 4, h));
  if (L.hair === "receding") (fill(3, 4, 4, 7, h), fill(11, 12, 4, 7, h), px(5, 3, h), px(10, 3, h));
  if (L.hair === "bald") (fill(3, 3, 6, 8, h), fill(12, 12, 6, 8, h));
  if (L.hair === "wild") {
    fill(3, 12, 1, 3, h);
    for (const [x, y] of [
      [1, 2],
      [2, 1],
      [14, 2],
      [13, 1],
      [2, 4],
      [13, 4],
      [1, 5],
      [14, 5],
      [3, 0],
      [7, 0],
      [11, 0],
    ])
      px(x, y, h);
    (fill(2, 3, 4, 7, h), fill(12, 13, 4, 7, h));
  }

  // facial hair
  const b = L.beardColor ?? h;
  if (L.beard === "mustache") fill(6, 9, 8, 8, b);
  if (L.beard === "goatee") (fill(6, 9, 8, 8, b), fill(7, 8, 10, 11, b));
  if (L.beard === "full") (fill(4, 11, 8, 11, b), fill(5, 10, 12, 12, b), fill(7, 8, 9, 9, "#9a5b4b"));
  if (L.beard === "sideburns") (fill(4, 4, 5, 9, b), fill(11, 11, 5, 9, b));

  // glasses
  if (L.glasses) {
    for (const cx of [6, 9]) {
      if (L.glasses === "square") (fill(cx - 1, cx + 1, 6, 6, INK), fill(cx - 1, cx + 1, 8, 8, INK));
      else (px(cx, 6, INK), px(cx, 8, INK));
      px(cx - 1, 7, INK);
      px(cx + 1, 7, INK);
    }
  }

  // prop at the lower right
  if (L.prop === "box")
    (fill(11, 15, 10, 15, "#c8a96a"), fill(12, 14, 11, 11, "#a88a4f"), px(13, 13, INK), px(13, 12, INK));
  if (L.prop === "apple")
    (fill(12, 14, 11, 13, "#b5443c"), px(13, 10, "#4d7c3a"), px(14, 14, "#b5443c"), px(12, 14, "#b5443c"));
  if (L.prop === "cube")
    (fill(12, 15, 11, 14, "#93a3c4"), fill(13, 15, 10, 10, "#b8c4dd"), fill(15, 15, 10, 13, "#6f80a6"));
  if (L.prop === "tape") (fill(11, 15, 12, 13, CREAM), [11, 13, 15].forEach((x) => px(x, 12, INK)));
  if (L.prop === "flask") (fill(13, 13, 9, 11, GREY), fill(12, 14, 12, 14, "#7fb3a8"));
  return g;
}
