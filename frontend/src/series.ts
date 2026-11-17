/**
 * The learnkit series: the sister sites built on this shared layer, listed on every site's Further reading
 * (#/reading/labs and #/reading/explorables). One source of truth: Tensorizer and Clifford keep a copy of this file
 * until they adopt learnkit (then they import it). Text is keyed by language code, with English as the fallback, so
 * sites with more languages (Tensorizer has ten) don't fall back to English.
 *
 * - Code labs: you learn by writing code that a machine checks (Tensorizer: tests; Formalizer: the Lean kernel).
 * - Explorables: interactive books ("explorable explanations") you read and play with; nothing to install.
 */

export type SeriesKind = "lab" | "explorable";
/** text per language code ("en", "zh-Hant", …); "en" is required */
export type Texts = { en: string } & Record<string, string>;

export interface SeriesSite {
  id: "tensorizer" | "formalizer" | "clifford" | "polyphonizer" | "world-models";
  name: string;
  kind: SeriesKind;
  url: string;
  logo: string;
  /** languages the site offers (a link passes ?lang= only for these) */
  langs: string[];
  /** the site's colour, as literal Tailwind classes: a badge and a border */
  badge: string;
  border: string;
  pitch: Texts;
  /** what you do there, in a few words */
  verb: Texts;
}

export interface SeriesGroup {
  kind: SeriesKind;
  title: Texts;
  intro: Texts;
}

export const SERIES_TITLE: Texts = {
  en: "The learnkit series",
  "zh-Hant": "learnkit 系列",
  "zh-Hans": "learnkit 系列",
  fr: "La série learnkit",
  de: "Die learnkit-Reihe",
  es: "La serie learnkit",
  ja: "learnkit シリーズ",
  ru: "Серия learnkit",
  ko: "learnkit 시리즈",
  hi: "learnkit श्रृंखला",
};

export const SERIES_INTRO: Texts = {
  en: "Sister sites that share one design and one idea: learn a subject by doing it, with something that tells you honestly whether you got it right.",
  "zh-Hant": "共用同一套設計與同一個理念的姊妹網站：動手做來學一門學科，並有東西誠實地告訴你做對了沒有。",
  "zh-Hans": "共用同一套设计与同一个理念的姐妹网站：动手做来学一门学科，并有东西诚实地告诉你做对了没有。",
  fr: "Des sites frères qui partagent un même design et une même idée : apprendre une matière en la pratiquant, avec quelque chose qui vous dit honnêtement si vous avez juste.",
  de: "Schwesterseiten mit demselben Design und derselben Idee: ein Fach lernen, indem man es tut, mit etwas, das einem ehrlich sagt, ob es stimmt.",
  es: "Sitios hermanos que comparten un mismo diseño y una misma idea: aprender una materia haciéndola, con algo que te dice honestamente si lo hiciste bien.",
  ja: "同じデザインと同じ考え方を共有する姉妹サイト：実際に手を動かして学び、正しくできたかを正直に教えてくれる仕組みがある。",
  ru: "Родственные сайты с общим дизайном и общей идеей: изучать предмет, занимаясь им, и получать честный ответ, верно ли получилось.",
  ko: "같은 디자인과 같은 생각을 공유하는 자매 사이트: 직접 해 보며 배우고, 제대로 했는지 정직하게 알려 주는 장치가 있습니다.",
  hi: "एक ही डिज़ाइन और एक ही विचार वाली सहयोगी साइटें: किसी विषय को करके सीखिए, और कुछ ऐसा जो ईमानदारी से बताए कि आपने सही किया या नहीं।",
};

export const SERIES_GROUPS: SeriesGroup[] = [
  {
    kind: "lab",
    title: {
      en: "Code labs",
      "zh-Hant": "程式實驗室",
      "zh-Hans": "编程实验室",
      fr: "Laboratoires de code",
      de: "Code-Labore",
      es: "Laboratorios de código",
      ja: "コードラボ",
      ru: "Лаборатории кода",
      ko: "코드 랩",
      hi: "कोड लैब",
    },
    intro: {
      en: "You learn by writing code that a machine checks: tests for physics, the Lean kernel for proofs.",
      "zh-Hant": "動手寫程式、讓機器檢查來學習：物理交給測試，證明交給 Lean 的核心。",
      "zh-Hans": "动手写程序、让机器检查来学习：物理交给测试，证明交给 Lean 的内核。",
      fr: "On apprend en écrivant du code qu'une machine vérifie : des tests pour la physique, le noyau de Lean pour les preuves.",
      de: "Man lernt, indem man Code schreibt, den eine Maschine prüft: Tests für die Physik, Leans Kernel für Beweise.",
      es: "Se aprende escribiendo código que una máquina comprueba: pruebas para la física, el núcleo de Lean para las demostraciones.",
      ja: "機械が確かめるコードを書いて学ぶ：物理はテストで、証明は Lean のカーネルで。",
      ru: "Учитесь, сочиняя код, который проверяет машина: тесты для физики, ядро Lean для доказательств.",
      ko: "기계가 확인하는 코드를 직접 써 보며 배웁니다. 물리는 테스트가, 증명은 Lean 커널이 확인합니다.",
      hi: "ऐसा कोड लिखकर सीखिए जिसे मशीन जाँचती है: भौतिकी के लिए टेस्ट, प्रमाणों के लिए Lean का कर्नेल।",
    },
  },
  {
    kind: "explorable",
    title: {
      en: "Explorables",
      "zh-Hant": "互動讀本",
      "zh-Hans": "互动读本",
      fr: "Explorables",
      de: "Explorables",
      es: "Explorables",
      ja: "エクスプロラブル",
      ru: "Интерактивные книги",
      ko: "익스플로러블",
      hi: "एक्सप्लोरेबल्स",
    },
    intro: {
      en: "Interactive books: read, drag, and watch the ideas move. Nothing to install.",
      "zh-Hant": "互動式的書：邊讀邊拖曳，看著概念動起來。不需安裝任何東西。",
      "zh-Hans": "互动式的书：边读边拖动，看着概念动起来。无需安装任何东西。",
      fr: "Des livres interactifs : lisez, faites glisser, regardez les idées bouger. Rien à installer.",
      de: "Interaktive Bücher: lesen, ziehen und zusehen, wie sich die Ideen bewegen. Nichts zu installieren.",
      es: "Libros interactivos: lee, arrastra y mira cómo se mueven las ideas. Nada que instalar.",
      ja: "インタラクティブな本：読んで、ドラッグして、アイデアが動くのを見る。インストール不要。",
      ru: "Читайте, перетаскивайте и смотрите, как оживают идеи. Ничего не нужно устанавливать.",
      ko: "인터랙티브 책: 읽고, 끌어 보고, 아이디어가 움직이는 것을 보세요. 설치할 것은 없습니다.",
      hi: "इंटरैक्टिव किताबें: पढ़िए, खींचिए और विचारों को चलते देखिए। कुछ भी इंस्टॉल नहीं करना।",
    },
  },
];

export const SERIES_SITES: SeriesSite[] = [
  {
    id: "tensorizer",
    name: "Tensorizer",
    kind: "lab",
    url: "https://jackyko1991.github.io/tensorizer/",
    logo: "https://jackyko1991.github.io/tensorizer/favicon.svg",
    langs: ["en", "zh-Hant", "zh-Hans", "fr", "de", "es", "ja", "ru", "ko", "hi"],
    badge: "bg-blue-50 text-blue-800",
    border: "border-blue-200 hover:border-blue-400",
    pitch: {
      en: "Physics as tensor code: laws become PyTorch programs, checked by tests written first.",
      "zh-Hant": "用張量程式寫物理：定律變成 PyTorch 程式，由先寫好的測試檢查。",
      "zh-Hans": "用张量程序写物理：定律变成 PyTorch 程序，由先写好的测试检查。",
      fr: "La physique en code tensoriel : les lois deviennent des programmes PyTorch, vérifiés par des tests écrits d'abord.",
      de: "Physik als Tensor-Code: Gesetze werden zu PyTorch-Programmen, geprüft von zuerst geschriebenen Tests.",
      es: "La física como código de tensores: las leyes se convierten en programas de PyTorch, comprobados por pruebas escritas primero.",
      ja: "テンソルのコードで物理を：法則を PyTorch のプログラムにし、先に書いたテストで確かめる。",
      ru: "Физика как тензорный код: законы становятся программами на PyTorch и проверяются заранее написанными тестами.",
      ko: "텐서 코드로 배우는 물리: 법칙을 PyTorch 프로그램으로 만들고, 먼저 작성한 테스트로 확인합니다.",
      hi: "टेंसर कोड के रूप में भौतिकी: नियम PyTorch प्रोग्राम बनते हैं, जिन्हें पहले लिखे टेस्ट जाँचते हैं।",
    },
    verb: {
      en: "Physics · PyTorch",
      "zh-Hant": "物理 · PyTorch",
      "zh-Hans": "物理 · PyTorch",
      fr: "Physique · PyTorch",
      de: "Physik · PyTorch",
      es: "Física · PyTorch",
      ja: "物理 · PyTorch",
      ru: "Физика · PyTorch",
      ko: "물리 · PyTorch",
      hi: "भौतिकी · PyTorch",
    },
  },
  {
    id: "formalizer",
    name: "Formalizer",
    kind: "lab",
    url: "https://jackyko1991.github.io/formalizer/",
    // bundled here so the card shows the current logo without depending on that site's deploy
    logo: "series/formalizer.svg",
    langs: ["en", "zh-Hant"],
    badge: "bg-teal-50 text-teal-800",
    border: "border-teal-200 hover:border-teal-400",
    pitch: {
      en: "Mathematics in Lean 4: the statement is the test, the kernel is the grader. Runs on your own computer.",
      "zh-Hant": "用 Lean 4 學數學：陳述就是測試，核心負責閱卷。在你自己的電腦上執行。",
      "zh-Hans": "用 Lean 4 学数学：陈述就是测试，内核负责阅卷。在你自己的电脑上运行。",
      fr: "Les mathématiques en Lean 4 : l'énoncé est le test, le noyau est l'examinateur. Tourne sur votre ordinateur.",
      de: "Mathematik in Lean 4: Die Aussage ist der Test, der Kernel bewertet. Läuft auf Ihrem eigenen Rechner.",
      es: "Matemáticas en Lean 4: el enunciado es la prueba y el núcleo la corrige. Se ejecuta en tu propio ordenador.",
      ja: "Lean 4 で数学を：命題がテスト、カーネルが採点者。自分のコンピューターで動く。",
      ru: "Математика на Lean 4: утверждение — это тест, ядро — экзаменатор. Работает на вашем компьютере.",
      ko: "Lean 4로 배우는 수학: 명제가 곧 테스트이고 커널이 채점합니다. 내 컴퓨터에서 실행됩니다.",
      hi: "Lean 4 में गणित: कथन ही टेस्ट है, कर्नेल परीक्षक है। आपके अपने कंप्यूटर पर चलता है।",
    },
    verb: {
      en: "Proofs · Lean 4",
      "zh-Hant": "證明 · Lean 4",
      "zh-Hans": "证明 · Lean 4",
      fr: "Preuves · Lean 4",
      de: "Beweise · Lean 4",
      es: "Demostraciones · Lean 4",
      ja: "証明 · Lean 4",
      ru: "Доказательства · Lean 4",
      ko: "증명 · Lean 4",
      hi: "प्रमाण · Lean 4",
    },
  },
  {
    id: "polyphonizer",
    name: "Polyphonizer",
    kind: "lab",
    url: "https://jackyko1991.github.io/polyphonizer/",
    logo: "series/polyphonizer.svg",
    langs: ["en", "zh-Hant"],
    badge: "bg-rose-50 text-rose-800",
    border: "border-rose-200 hover:border-rose-400",
    pitch: {
      en: "Music scores written as code: counterpoint and arranging graded by rules, played back, and laid out with page turns at rests. In your browser.",
      "zh-Hant": "以程式碼寫樂譜：對位與編曲由規則評分，可播放，並在休止處翻頁排版。在瀏覽器中執行。",
      "zh-Hans": "以代码写乐谱：对位与编曲由规则评分，可播放，并在休止处翻页排版。在浏览器中运行。",
      fr: "Des partitions écrites en code : contrepoint et arrangement notés par des règles, joués, et mis en page avec les tournes sur les silences. Dans votre navigateur.",
      de: "Noten als Code: Kontrapunkt und Arrangement nach Regeln bewertet, abgespielt und mit Seitenwechseln in Pausen gesetzt. Im Browser.",
      es: "Partituras escritas como código: contrapunto y arreglos calificados por reglas, reproducidos y maquetados con los pases de página en los silencios. En tu navegador.",
      ja: "コードで書く楽譜：対位法と編曲をルールで採点し、再生し、休符でページをめくれるようにレイアウト。ブラウザで動く。",
      ru: "Ноты как код: контрапункт и аранжировка проверяются правилами, звучат и вёрстаются с перелистыванием на паузах. В браузере.",
      ko: "코드로 쓰는 악보: 대위법과 편곡을 규칙으로 채점하고, 재생하고, 쉼표에서 페이지를 넘기도록 배치합니다. 브라우저에서 실행됩니다.",
      hi: "कोड के रूप में लिखे संगीत स्कोर: काउंटरपॉइंट और अरेंजमेंट नियमों से जाँचे जाते हैं, बजाए जाते हैं, और विराम पर पन्ना पलटने के हिसाब से सजाए जाते हैं। आपके ब्राउज़र में।",
    },
    verb: {
      en: "Scores · music21",
      "zh-Hant": "樂譜 · music21",
      "zh-Hans": "乐谱 · music21",
      fr: "Partitions · music21",
      de: "Noten · music21",
      es: "Partituras · music21",
      ja: "楽譜 · music21",
      ru: "Ноты · music21",
      ko: "악보 · music21",
      hi: "स्कोर · music21",
    },
  },
  {
    id: "clifford",
    name: "Clifford Algebra",
    kind: "explorable",
    url: "https://jackyko1991.github.io/clifford/",
    logo: "https://jackyko1991.github.io/clifford/icon.svg",
    langs: ["en", "zh-Hant"],
    badge: "bg-violet-50 text-violet-800",
    border: "border-violet-200 hover:border-violet-400",
    pitch: {
      en: "Geometric algebra through diagrams, animation, equations and code: rotations, rotors and spinors you can play with.",
      "zh-Hant": "用圖、動畫、方程式與程式碼學幾何代數：可以動手玩的旋轉、轉子與旋量。",
      "zh-Hans": "用图、动画、方程与代码学几何代数：可以动手玩的旋转、转子与旋量。",
      fr: "L'algèbre géométrique par les diagrammes, l'animation, les équations et le code : rotations, rotors et spineurs à manipuler.",
      de: "Geometrische Algebra mit Diagrammen, Animationen, Gleichungen und Code: Drehungen, Rotoren und Spinoren zum Ausprobieren.",
      es: "Álgebra geométrica con diagramas, animación, ecuaciones y código: rotaciones, rotores y espinores para experimentar.",
      ja: "図・アニメーション・数式・コードで学ぶ幾何代数：触って試せる回転、ローター、スピノル。",
      ru: "Геометрическая алгебра в диаграммах, анимации, формулах и коде: вращения, роторы и спиноры, с которыми можно поиграть.",
      ko: "도형, 애니메이션, 수식, 코드로 배우는 기하 대수: 직접 만져 볼 수 있는 회전, 로터, 스피너.",
      hi: "आरेख, एनिमेशन, समीकरण और कोड से ज्यामितीय बीजगणित: घूर्णन, रोटर और स्पिनर जिनके साथ आप खेल सकें।",
    },
    verb: {
      en: "Geometric algebra",
      "zh-Hant": "幾何代數",
      "zh-Hans": "几何代数",
      fr: "Algèbre géométrique",
      de: "Geometrische Algebra",
      es: "Álgebra geométrica",
      ja: "幾何代数",
      ru: "Геометрическая алгебра",
      ko: "기하 대수",
      hi: "ज्यामितीय बीजगणित",
    },
  },
  {
    id: "world-models",
    name: "World Models",
    kind: "explorable",
    url: "https://jackyko1991.github.io/world-models/",
    // bundled here so the card shows the logo before that site is deployed
    logo: "series/world-models.svg",
    langs: ["en", "zh-Hant"],
    badge: "bg-amber-50 text-amber-800",
    border: "border-amber-200 hover:border-amber-400",
    pitch: {
      en: "World models through diagrams, animation, equations and code: predict, imagine and plan, from Dyna and Dreamer to Genie, NeRF and Gaussian splatting.",
      "zh-Hant": "用圖、動畫、方程式與程式碼學世界模型：預測、想像與規劃，從 Dyna、Dreamer 到 Genie、NeRF 與高斯潑濺。",
      "zh-Hans": "用图、动画、方程与代码学世界模型：预测、想象与规划，从 Dyna、Dreamer 到 Genie、NeRF 与高斯泼溅。",
      fr: "Les modèles du monde par les diagrammes, l'animation, les équations et le code : prédire, imaginer et planifier, de Dyna et Dreamer à Genie, NeRF et au Gaussian splatting.",
      de: "Weltmodelle mit Diagrammen, Animationen, Gleichungen und Code: vorhersagen, sich vorstellen und planen, von Dyna und Dreamer bis Genie, NeRF und Gaussian Splatting.",
      es: "Modelos del mundo con diagramas, animación, ecuaciones y código: predecir, imaginar y planificar, de Dyna y Dreamer a Genie, NeRF y el Gaussian splatting.",
      ja: "図・アニメーション・数式・コードで学ぶ世界モデル：予測し、想像し、計画する。Dyna や Dreamer から Genie、NeRF、ガウシアン・スプラッティングまで。",
      ru: "Модели мира в диаграммах, анимации, формулах и коде: предсказывать, воображать и планировать, от Dyna и Dreamer до Genie, NeRF и гауссова сплаттинга.",
      ko: "도형, 애니메이션, 수식, 코드로 배우는 월드 모델: 예측하고, 상상하고, 계획하기. Dyna와 Dreamer부터 Genie, NeRF, 가우시안 스플래팅까지.",
      hi: "आरेख, एनिमेशन, समीकरण और कोड से वर्ल्ड मॉडल: भविष्यवाणी, कल्पना और योजना, Dyna और Dreamer से Genie, NeRF और गॉसियन स्प्लैटिंग तक।",
    },
    verb: {
      en: "World models",
      "zh-Hant": "世界模型",
      "zh-Hans": "世界模型",
      fr: "Modèles du monde",
      de: "Weltmodelle",
      es: "Modelos del mundo",
      ja: "世界モデル",
      ru: "Модели мира",
      ko: "월드 모델",
      hi: "वर्ल्ड मॉडल",
    },
  },
];

/** The text in `lang`, or English. */
export const seriesText = (texts: Texts, lang: string): string => texts[lang] ?? texts.en;

/** A sister site's address in the reader's language when it has it (`?lang=`), else its default. */
export const seriesUrl = (site: SeriesSite, lang: string): string =>
  site.langs.includes(lang) ? `${site.url}?lang=${encodeURIComponent(lang)}` : site.url;

/** Labels for the Further reading tabs on every site of the series (books, labs, explorables). */
export const READING_TABS: Record<"books" | "labs" | "explorables", Texts> = {
  books: {
    en: "Books & papers",
    "zh-Hant": "書籍與論文",
    "zh-Hans": "书籍与论文",
    fr: "Livres et articles",
    de: "Bücher und Artikel",
    es: "Libros y artículos",
    ja: "本と論文",
    ru: "Книги и статьи",
    ko: "책과 논문",
    hi: "पुस्तकें और शोध-पत्र",
  },
  labs: SERIES_GROUPS[0].title,
  explorables: SERIES_GROUPS[1].title,
};

export const YOU_ARE_HERE: Texts = {
  en: "You are here",
  "zh-Hant": "你在這裡",
  "zh-Hans": "你在这里",
  fr: "Vous êtes ici",
  de: "Sie sind hier",
  es: "Estás aquí",
  ja: "現在のサイト",
  ru: "Вы здесь",
  ko: "현재 위치",
  hi: "आप यहाँ हैं",
};

export const VISIT: Texts = {
  en: "Open",
  "zh-Hant": "前往",
  "zh-Hans": "前往",
  fr: "Ouvrir",
  de: "Öffnen",
  es: "Abrir",
  ja: "開く",
  ru: "Открыть",
  ko: "열기",
  hi: "खोलें",
};
