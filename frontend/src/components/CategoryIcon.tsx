/** A small line icon for each topic of the problem library (24×24 viewBox, stroke = current colour). */
const PATHS: Record<string, React.ReactNode> = {
  // pendulum: pivot, rod, bob, swing arc
  mechanics: (
    <>
      <path d="M6 3h12M12 3l4 12" />
      <circle cx="16.5" cy="17" r="2.5" />
      <path d="M8 15a8 8 0 0 0 4 3" strokeDasharray="1.5 2" />
    </>
  ),
  // prism splitting a ray
  optics: (
    <>
      <path d="M12 4l7 14H5z" />
      <path d="M2 13l7-1M15 12l7-3M15 13l7 0M15 14l7 3" />
    </>
  ),
  // two waves
  waves: <path d="M2 9c2.5-4 5-4 7.5 0s5 4 7.5 0 3.5-3 5-2M2 16c2.5-4 5-4 7.5 0s5 4 7.5 0 3.5-3 5-2" />,
  // thermometer
  thermodynamics: (
    <>
      <path d="M10 14V5a2 2 0 0 1 4 0v9a4 4 0 1 1-4 0z" />
      <path d="M12 9v7" />
    </>
  ),
  // lightning bolt
  em: <path d="M13 2L5 13h6l-1 9 8-11h-6z" />,
  // droplet with flow line
  fluids: (
    <>
      <path d="M12 3c3.5 4.5 6 7.5 6 11a6 6 0 0 1-12 0c0-3.5 2.5-6.5 6-11z" />
      <path d="M9 15a3 3 0 0 0 3 3" />
    </>
  ),
  // atom: nucleus and two orbits
  quantum: (
    <>
      <circle cx="12" cy="12" r="1.6" />
      <ellipse cx="12" cy="12" rx="10" ry="4" />
      <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)" />
      <ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)" />
    </>
  ),
  // ringed planet and a star
  astro: (
    <>
      <circle cx="11" cy="13" r="5" />
      <ellipse cx="11" cy="13" rx="10" ry="3" transform="rotate(-20 11 13)" />
      <path d="M19 3v4M17 5h4" />
    </>
  ),
  // galaxy spiral and a distant star: cosmology
  cosmology: (
    <>
      <path d="M12 12c0-1.5 1.5-2.5 3-2 2 .7 2.5 3.3 1 5-2 2.3-6 2-7.5-.5C6.6 11.3 8.5 6.5 13 6c4.5-.5 8 3.5 7 8" />
      <path d="M4 5l.6 1.4L6 7l-1.4.6L4 9l-.6-1.4L2 7l1.4-.6z" />
    </>
  ),
  // radiation trefoil
  nuclear: (
    <>
      <circle cx="12" cy="12" r="1.8" />
      <path d="M12 9.5L9 4.3a8 8 0 0 1 6 0zM14.2 13.3l6 .1a8 8 0 0 1-3 5.2zM9.8 13.3l-3 5.3a8 8 0 0 1-3-5.2z" />
    </>
  ),
  // conical flask
  physchem: (
    <>
      <path d="M9 3h6M10 3v6l-5.5 9.5A1.6 1.6 0 0 0 5.9 21h12.2a1.6 1.6 0 0 0 1.4-2.5L14 9V3" />
      <path d="M7.5 15h9" />
    </>
  ),
  // benzene ring
  chemistry: (
    <>
      <path d="M12 3l7.8 4.5v9L12 21l-7.8-4.5v-9z" />
      <circle cx="12" cy="12" r="4" />
    </>
  ),
  // heart with pulse
  medical: (
    <>
      <path d="M12 20s-8-4.8-8-10.5A4.5 4.5 0 0 1 12 7a4.5 4.5 0 0 1 8 2.5C20 15.2 12 20 12 20z" />
      <path d="M4.5 12h4l1.5-3 2.5 6 1.5-3h5.5" />
    </>
  ),
  // DNA double helix
  biophysics: (
    <>
      <path d="M7 3c0 6 10 6 10 12s-10 3-10 6M17 3c0 6-10 6-10 12s10 3 10 6" />
      <path d="M9 6.5h6M9 17.5h6" />
    </>
  ),
  // sequence bars (alignment)
  bioinformatics: (
    <>
      <path d="M3 6h4M9 6h12M3 12h9M14 12h7M3 18h6M11 18h10" />
    </>
  ),
  // sigma
  maths: <path d="M18 5H6l6 7-6 7h12" />,
  // controller-free: a cube on a pixel grid
  graphics: (
    <>
      <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" />
      <path d="M4 7.5l8 4.5 8-4.5M12 12v9" />
    </>
  ),
  // a small network: three layers of nodes
  ai: (
    <>
      <circle cx="5" cy="7" r="1.6" />
      <circle cx="5" cy="17" r="1.6" />
      <circle cx="12" cy="5" r="1.6" />
      <circle cx="12" cy="12" r="1.6" />
      <circle cx="12" cy="19" r="1.6" />
      <circle cx="19" cy="12" r="1.6" />
      <path d="M6.5 7.5l4 -2M6.5 7.8l4 3.6M6.5 16.2l4 -3.6M6.5 16.5l4 2M13.6 5.6l3.9 5.4M13.6 12h3.8M13.6 18.4l3.9 -5.4" />
    </>
  ),
  // sparkles
  others: (
    <>
      <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />
      <path d="M19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z" />
    </>
  ),
};

export default function CategoryIcon({ id, className = "h-4 w-4" }: { id: string; className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {PATHS[id] ?? PATHS.others}
    </svg>
  );
}
