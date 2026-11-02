import { useEffect, type ReactNode } from "react";

export interface MenuItem {
  label: string;
  icon?: ReactNode;
  /** an action, or a link opened in a new tab */
  onClick?: () => void;
  href?: string;
}

/** A right-click menu at the pointer; closes on any click, scroll or key. */
export default function ContextMenu({
  x,
  y,
  title,
  items,
  onClose,
}: {
  x: number;
  y: number;
  title?: string;
  items: MenuItem[];
  onClose: () => void;
}) {
  useEffect(() => {
    const close = () => onClose();
    window.addEventListener("click", close);
    window.addEventListener("keydown", close);
    window.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", close);
      window.removeEventListener("scroll", close, true);
    };
  }, [onClose]);

  const left = Math.min(x, window.innerWidth - 240);
  const top = Math.min(y, window.innerHeight - 40 - items.length * 32);
  const row =
    "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-xs text-gray-700 hover:bg-blue-50";
  return (
    <div
      role="menu"
      style={{ left, top }}
      className="fixed z-50 w-56 rounded-lg border border-gray-200 bg-white p-1 shadow-lg"
      data-testid="context-menu"
    >
      {title && <p className="truncate px-2.5 py-1 text-[11px] text-gray-400">{title}</p>}
      {items.map((item) =>
        item.href ? (
          <a key={item.label} role="menuitem" href={item.href} target="_blank" rel="noreferrer" className={row}>
            <span className="w-4 text-center">{item.icon}</span>
            {item.label}
          </a>
        ) : (
          <button key={item.label} role="menuitem" onClick={item.onClick} className={row}>
            <span className="w-4 text-center">{item.icon}</span>
            {item.label}
          </button>
        ),
      )}
    </div>
  );
}
