import { python } from "@codemirror/lang-python";
import { EditorView } from "@codemirror/view";
import CodeMirror from "@uiw/react-codemirror";

const EXTENSIONS = [python(), EditorView.lineWrapping]; // long lines wrap: no sideways scrolling

/** Read-only, syntax-highlighted Python snippet. */
export default function CodeBlock({ code, title }: { code: string; title?: string }) {
  return (
    <figure className="overflow-hidden rounded-lg border border-gray-200 bg-white">
      {title && (
        <figcaption className="border-b border-gray-200 bg-gray-50 px-3 py-1.5 font-mono text-xs text-gray-600">
          {title}
        </figcaption>
      )}
      <CodeMirror
        value={code.trimEnd()}
        extensions={EXTENSIONS}
        editable={false}
        readOnly
        theme="light"
        basicSetup={{ foldGutter: false, highlightActiveLine: false, lineNumbers: false }}
      />
    </figure>
  );
}

/** The source of one top-level `def name(...)` from a Python file (up to the next top-level statement). */
export function extractDef(source: string, name: string): string {
  const lines = source.split("\n");
  const start = lines.findIndex((l) => l.startsWith(`def ${name}(`));
  if (start < 0) return `# ${name}() not found`;
  let end = start + 1;
  while (end < lines.length && (lines[end] === "" || /^\s/.test(lines[end]))) end++;
  return lines.slice(start, end).join("\n").trimEnd();
}
