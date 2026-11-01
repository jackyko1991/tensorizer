/** The agent's face: Tensorizer's 3×3×3 tensor block with a friendly emoji face. It blinks while idle and looks up
 *  with a "…" bubble while the agent is thinking. */
export default function AgentMascot({ thinking = false, className = "" }: { thinking?: boolean; className?: string }) {
  const pupilY = thinking ? 33 : 36;
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden="true">
      <image href={`${import.meta.env.BASE_URL}favicon.svg`} x="0" y="0" width="64" height="64" />
      {/* face on the two front faces of the cube */}
      <g className="mascot-eyes">
        <ellipse cx="23" cy="35" rx="5" ry="6" fill="#ffffff" />
        <ellipse cx="41" cy="35" rx="5" ry="6" fill="#ffffff" />
        <circle cx={thinking ? 24 : 23.5} cy={pupilY} r="2.6" fill="#111827" />
        <circle cx={thinking ? 42 : 41.5} cy={pupilY} r="2.6" fill="#111827" />
      </g>
      {thinking ? (
        <ellipse cx="32" cy="47" rx="3" ry="2.2" fill="#111827" />
      ) : (
        <path d="M24 45 Q32 52 40 45" fill="none" stroke="#111827" strokeWidth="2.4" strokeLinecap="round" />
      )}
      <circle cx="17" cy="43" r="2.2" fill="#f9a8d4" opacity="0.8" />
      <circle cx="47" cy="43" r="2.2" fill="#f9a8d4" opacity="0.8" />
      {thinking && (
        <g className="mascot-dots" fill="#f59e0b">
          <circle cx="50" cy="8" r="2.4" />
          <circle cx="56" cy="8" r="2.4" />
          <circle cx="62" cy="8" r="2.4" />
        </g>
      )}
    </svg>
  );
}
