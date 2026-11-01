/**
 * Interface translations and the language context.
 *
 * UI strings live here. Problem content (title, summary, notes, meanings, parameter labels) comes from
 * problems/<category>/<slug>/i18n/<lang>.yaml via the manifest and falls back to English.
 */
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import type { Problem } from "./types";

export const LANGS = [
  { code: "en", name: "English" },
  { code: "zh-Hant", name: "繁體中文" },
  { code: "zh-Hans", name: "简体中文" },
  { code: "fr", name: "Français" },
  { code: "de", name: "Deutsch" },
  { code: "es", name: "Español" },
  { code: "ja", name: "日本語" },
  { code: "ru", name: "Русский" },
  { code: "ko", name: "한국어" },
] as const;
export type Lang = (typeof LANGS)[number]["code"];

// One row per key: en, zh-Hant, zh-Hans, fr, de, es, ja, ru, ko
const TABLE = {
  tagline: ["Physics equations solver by GPU tensors", "以 GPU 張量求解物理方程", "用 GPU 张量求解物理方程", "Résolution d'équations physiques par tenseurs GPU", "Physikalische Gleichungen mit GPU-Tensoren lösen", "Resolución de ecuaciones físicas con tensores en GPU", "GPUテンソルで物理方程式を解く", "Решение физических уравнений на тензорах GPU", "GPU 텐서로 푸는 물리 방정식"],
  language: ["Language", "語言", "语言", "Langue", "Sprache", "Idioma", "言語", "Язык", "언어"],
  "backend.checking": ["checking backend…", "正在檢查後端…", "正在检查后端…", "vérification du serveur…", "Backend wird geprüft…", "comprobando el servidor…", "バックエンドを確認中…", "проверка сервера…", "백엔드 확인 중…"],
  "backend.edit": ["backend · code editing on", "後端 · 可編輯程式碼", "后端 · 可编辑代码", "serveur · édition du code activée", "Backend · Code bearbeitbar", "servidor · edición de código activada", "バックエンド · コード編集可", "сервер · редактирование кода включено", "백엔드 · 코드 편집 가능"],
  "backend.online": ["backend online", "後端已連線", "后端已连接", "serveur en ligne", "Backend online", "servidor en línea", "バックエンド接続中", "сервер в сети", "백엔드 연결됨"],
  "backend.offline": ["backend offline · prebuilt results only", "後端離線 · 僅顯示預先產生的結果", "后端离线 · 仅显示预生成结果", "serveur hors ligne · résultats précalculés uniquement", "Backend offline · nur vorberechnete Ergebnisse", "servidor sin conexión · solo resultados precalculados", "バックエンド未接続 · 事前生成の結果のみ", "сервер недоступен · только готовые результаты", "백엔드 오프라인 · 미리 생성된 결과만"],
  library: ["Problem library", "題庫", "题库", "Bibliothèque de problèmes", "Aufgabensammlung", "Biblioteca de problemas", "問題ライブラリ", "Библиотека задач", "문제 라이브러리"],
  "library.fold": ["Fold library", "收合題庫", "收起题库", "Replier la bibliothèque", "Sammlung einklappen", "Plegar biblioteca", "ライブラリを折りたたむ", "Свернуть библиотеку", "라이브러리 접기"],
  "library.open": ["Open library", "展開題庫", "展开题库", "Ouvrir la bibliothèque", "Sammlung öffnen", "Abrir biblioteca", "ライブラリを開く", "Открыть библиотеку", "라이브러리 열기"],
  "level.beginner": ["Beginner", "入門", "入门", "Débutant", "Einsteiger", "Principiante", "初級", "Начальный", "초급"],
  "level.intermediate": ["Intermediate", "中級", "中级", "Intermédiaire", "Mittelstufe", "Intermedio", "中級", "Средний", "중급"],
  "level.advanced": ["Advanced", "進階", "进阶", "Avancé", "Fortgeschritten", "Avanzado", "上級", "Продвинутый", "고급"],
  "cat.mechanics": ["Mechanics", "力學", "力学", "Mécanique", "Mechanik", "Mecánica", "力学", "Механика", "역학"],
  "cat.optics": ["Optics", "光學", "光学", "Optique", "Optik", "Óptica", "光学", "Оптика", "광학"],
  "cat.waves": ["Waves", "波動", "波动", "Ondes", "Wellen", "Ondas", "波動", "Волны", "파동"],
  "cat.thermodynamics": ["Thermodynamics", "熱力學", "热力学", "Thermodynamique", "Thermodynamik", "Termodinámica", "熱力学", "Термодинамика", "열역학"],
  "cat.em": ["Electromagnetism", "電磁學", "电磁学", "Électromagnétisme", "Elektromagnetismus", "Electromagnetismo", "電磁気学", "Электромагнетизм", "전자기학"],
  "cat.fluids": ["Fluids & Continuum", "流體與連續體", "流体与连续介质", "Fluides et milieux continus", "Fluide & Kontinuum", "Fluidos y medios continuos", "流体・連続体", "Жидкости и сплошные среды", "유체와 연속체"],
  "cat.quantum": ["Quantum", "量子力學", "量子力学", "Quantique", "Quantenmechanik", "Cuántica", "量子力学", "Квантовая механика", "양자역학"],
  "cat.astro": ["Astrophysics & Relativity", "天文物理與相對論", "天体物理与相对论", "Astrophysique et relativité", "Astrophysik & Relativität", "Astrofísica y relatividad", "天体物理・相対論", "Астрофизика и теория относительности", "천체물리와 상대론"],
  "cat.nuclear": ["Nuclear & Particle", "核與粒子物理", "核与粒子物理", "Physique nucléaire et des particules", "Kern- & Teilchenphysik", "Nuclear y de partículas", "原子核・素粒子", "Ядерная физика и частицы", "핵·입자물리"],
  "tag.animated": ["animated", "動畫", "动画", "animé", "animiert", "animado", "アニメーション", "анимация", "애니메이션"],
  "tag.static": ["static", "靜態", "静态", "statique", "statisch", "estático", "静止画", "статично", "정적"],
  setup: ["Physical setup", "物理情境", "物理情境", "Dispositif physique", "Physikalischer Aufbau", "Montaje físico", "物理的な設定", "Физическая постановка", "물리적 설정"],
  "eq.vector": ["Vector form", "向量形式", "向量形式", "Forme vectorielle", "Vektorform", "Forma vectorial", "ベクトル形式", "Векторная форма", "벡터 형식"],
  "eq.index": ["Index (tensor) form", "指標（張量）形式", "指标（张量）形式", "Forme indicielle (tensorielle)", "Indexform (Tensor)", "Forma indicial (tensorial)", "添字（テンソル）形式", "Индексная (тензорная) форма", "첨자(텐서) 형식"],
  "eq.translation": ["Translation to PyTorch", "轉換為 PyTorch", "转换为 PyTorch", "Traduction en PyTorch", "Übersetzung nach PyTorch", "Traducción a PyTorch", "PyTorch への変換", "Перевод в PyTorch", "PyTorch로 변환"],
  "col.symbol": ["Symbol", "符號", "符号", "Symbole", "Symbol", "Símbolo", "記号", "Символ", "기호"],
  "col.code": ["Code", "程式碼", "代码", "Code", "Code", "Código", "コード", "Код", "코드"],
  "col.shape": ["Shape", "形狀", "形状", "Forme", "Form", "Forma", "形状", "Форма", "형태"],
  "col.meaning": ["Meaning", "含義", "含义", "Signification", "Bedeutung", "Significado", "意味", "Смысл", "의미"],
  "role.main": ["main loop", "主迴圈", "主循环", "boucle principale", "Hauptschleife", "bucle principal", "メインループ", "основной цикл", "메인 루프"],
  "role.equation": ["physics → tensor translation", "物理 → 張量轉換", "物理 → 张量转换", "physique → tenseurs", "Physik → Tensoren", "física → tensores", "物理 → テンソル変換", "физика → тензоры", "물리 → 텐서 변환"],
  "role.tensors": ["PyTorch tensors", "PyTorch 張量", "PyTorch 张量", "tenseurs PyTorch", "PyTorch-Tensoren", "tensores PyTorch", "PyTorch テンソル", "тензоры PyTorch", "PyTorch 텐서"],
  "role.solve": ["solver", "求解器", "求解器", "solveur", "Löser", "resolvedor", "ソルバー", "решатель", "솔버"],
  "role.plot": ["2D/3D plot → GIF", "2D/3D 繪圖 → GIF", "2D/3D 绘图 → GIF", "tracé 2D/3D → GIF", "2D/3D-Plot → GIF", "gráfico 2D/3D → GIF", "2D/3D プロット → GIF", "график 2D/3D → GIF", "2D/3D 그래프 → GIF"],
  "code.editable": ["editable", "可編輯", "可编辑", "modifiable", "bearbeitbar", "editable", "編集可", "редактируемый", "편집 가능"],
  "code.readonly": ["read-only", "唯讀", "只读", "lecture seule", "schreibgeschützt", "solo lectura", "読み取り専用", "только чтение", "읽기 전용"],
  "code.reset": ["reset all", "全部重設", "全部重置", "tout réinitialiser", "alles zurücksetzen", "restablecer todo", "すべて元に戻す", "сбросить всё", "모두 초기화"],
  "params.title": ["Parameters", "參數", "参数", "Paramètres", "Parameter", "Parámetros", "パラメータ", "Параметры", "매개변수"],
  "params.open": ["Open parameters", "展開參數", "展开参数", "Ouvrir les paramètres", "Parameter öffnen", "Abrir parámetros", "パラメータを開く", "Открыть параметры", "매개변수 열기"],
  "params.fold": ["Fold parameters", "收合參數", "收起参数", "Replier les paramètres", "Parameter einklappen", "Plegar parámetros", "パラメータを折りたたむ", "Свернуть параметры", "매개변수 접기"],
  "params.verbose": ["Verbose output", "詳細輸出", "详细输出", "Sortie détaillée", "Ausführliche Ausgabe", "Salida detallada", "詳細出力", "Подробный вывод", "자세한 출력"],
  "params.verboseHint": ["Trace each step with tensor shapes, timings and progress bars", "逐步追蹤張量形狀、耗時與進度條", "逐步跟踪张量形状、耗时和进度条", "Trace chaque étape : formes des tenseurs, durées et barres de progression", "Jeden Schritt mit Tensorformen, Laufzeiten und Fortschrittsbalken verfolgen", "Traza cada paso con formas de tensores, tiempos y barras de progreso", "各ステップのテンソル形状・所要時間・進捗バーを表示", "Трассировка шагов: формы тензоров, время и индикаторы выполнения", "각 단계의 텐서 형태, 소요 시간, 진행 막대를 표시"],
  "params.defaults": ["Default Param", "預設參數", "默认参数", "Paramètres par défaut", "Standardwerte", "Parámetros por defecto", "デフォルト値", "По умолчанию", "기본값"],
  run: ["▶ Run", "▶ 執行", "▶ 运行", "▶ Exécuter", "▶ Ausführen", "▶ Ejecutar", "▶ 実行", "▶ Запуск", "▶ 실행"],
  "run.mine": ["▶ Run my code", "▶ 執行我的程式碼", "▶ 运行我的代码", "▶ Exécuter mon code", "▶ Meinen Code ausführen", "▶ Ejecutar mi código", "▶ 自分のコードを実行", "▶ Запустить мой код", "▶ 내 코드 실행"],
  "run.running": ["Running…", "執行中…", "运行中…", "Exécution…", "Läuft…", "Ejecutando…", "実行中…", "Выполняется…", "실행 중…"],
  "run.needBackend": ["Start the backend (pixi run dev) to run", "請先啟動後端（pixi run dev）才能執行", "请先启动后端（pixi run dev）才能运行", "Lancez le serveur (pixi run dev) pour exécuter", "Backend starten (pixi run dev), um auszuführen", "Inicia el servidor (pixi run dev) para ejecutar", "実行するにはバックエンドを起動してください（pixi run dev）", "Запустите сервер (pixi run dev), чтобы выполнить", "실행하려면 백엔드를 시작하세요 (pixi run dev)"],
  terminal: ["Terminal", "終端機", "终端", "Terminal", "Terminal", "Terminal", "ターミナル", "Терминал", "터미널"],
  plot: ["Plot", "圖表", "图表", "Graphique", "Plot", "Gráfico", "プロット", "График", "그래프"],
  "plot.rendering": ["rendering…", "繪製中…", "绘制中…", "rendu…", "wird gerendert…", "renderizando…", "描画中…", "отрисовка…", "렌더링 중…"],
  "plot.zoom": ["zoom", "放大", "放大", "agrandir", "vergrößern", "ampliar", "拡大", "увеличить", "확대"],
  "plot.none": ["No output yet — press Run.", "尚無輸出 — 請按執行。", "尚无输出 — 请点击运行。", "Pas encore de résultat — cliquez sur Exécuter.", "Noch keine Ausgabe — auf Ausführen klicken.", "Aún no hay resultado — pulsa Ejecutar.", "まだ出力がありません — 実行を押してください。", "Пока нет результата — нажмите «Запуск».", "아직 출력이 없습니다 — 실행을 누르세요."],
  "plot.close": ["click or Esc to close", "點擊或按 Esc 關閉", "点击或按 Esc 关闭", "cliquez ou Échap pour fermer", "Klicken oder Esc zum Schließen", "haz clic o pulsa Esc para cerrar", "クリックまたは Esc で閉じる", "щелкните или Esc, чтобы закрыть", "클릭하거나 Esc로 닫기"],
} satisfies Record<string, readonly string[]>;
export type Key = keyof typeof TABLE;

const STORAGE_KEY = "tensorizer.lang";

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && LANGS.some((l) => l.code === saved)) return saved as Lang;
  } catch {
    /* storage unavailable */
  }
  const nav = navigator.language.toLowerCase();
  if (nav.startsWith("zh")) return /tw|hk|mo|hant/.test(nav) ? "zh-Hant" : "zh-Hans";
  return (LANGS.find((l) => nav.startsWith(l.code))?.code ?? "en") as Lang;
}

interface LangState {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: Key) => string;
}

const LangContext = createContext<LangState | null>(null);

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    document.documentElement.lang = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* storage unavailable */
    }
  }, []);
  const value = useMemo<LangState>(() => {
    const index = LANGS.findIndex((l) => l.code === lang);
    return { lang, setLang, t: (key) => TABLE[key][index] ?? TABLE[key][0] };
  }, [lang, setLang]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang(): LangState {
  const state = useContext(LangContext);
  if (!state) throw new Error("useLang outside LangProvider");
  return state;
}

/** The problem with its text fields swapped for the chosen language where a translation exists. */
export function localize(problem: Problem, lang: Lang): Problem {
  const tr = problem.i18n?.[lang];
  if (!tr) return problem;
  return {
    ...problem,
    title: tr.title ?? problem.title,
    summary: tr.summary ?? problem.summary,
    notes: tr.notes ?? problem.notes,
    translation: problem.translation.map((row, i) => ({ ...row, meaning: tr.meanings?.[i] ?? row.meaning })),
    params: Object.fromEntries(
      Object.entries(problem.params).map(([name, spec]) => [name, { ...spec, label: tr.labels?.[name] ?? spec.label }]),
    ),
  };
}
