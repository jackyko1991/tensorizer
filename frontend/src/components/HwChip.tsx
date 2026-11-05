/**
 * Chip badge next to a card title: hovering shows `text` in the site-wide white HoverTip card (it turns any
 * `title` into that card and keeps the line breaks), under `label` as a bold heading.
 */
export default function HwChip({ label, text }: { label: string; text: string }) {
  return (
    <span className="inline-flex cursor-help self-center" title={text} data-tip-heading={label} aria-label={label} role="img" data-testid="hw-chip">
      <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden>
        <g stroke="#64748b" strokeWidth={1.6} strokeLinecap="round">
          {[8, 12, 16].map((v) => (
            <g key={v}>
              <line x1={v} y1={1.8} x2={v} y2={5} />
              <line x1={v} y1={19} x2={v} y2={22.2} />
              <line x1={1.8} y1={v} x2={5} y2={v} />
              <line x1={19} y1={v} x2={22.2} y2={v} />
            </g>
          ))}
        </g>
        <rect x={4.5} y={4.5} width={15} height={15} rx={2.5} fill="#334155" />
        <rect x={8} y={8} width={8} height={8} rx={1.2} fill="#60a5fa" />
        <circle cx={6.8} cy={6.8} r={0.9} fill="#cbd5e1" />
      </svg>
    </span>
  );
}
