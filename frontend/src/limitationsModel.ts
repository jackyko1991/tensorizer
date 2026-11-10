/** Joint outcomes in one common measurement basis; rows A, columns B. */
export function jointProbabilities(paired: boolean): number[][] {
  return paired ? [[.5,0],[0,.5]] : [[.25,.25],[.25,.25]];
}
/** The right loop cannot cross the excluded disk of radius 28. */
export function loopRadii(percent: number): [number,number] {
  const radius=51*(1-Math.min(100,Math.max(0,percent))/100);
  return [Math.max(0,radius),Math.max(32,radius)];
}

/** Display equations of the Limits of tensors tab, keyed by section; each section's list matches its eq1, eq2 labels. */
export const LIMITATION_EQUATIONS = {
  scope: [String.raw`v_{\mathrm{new}}=(\cos\theta,-\sin\theta),\qquad v'_i=R_{ij}\,v_j`],
  spinors: [
    String.raw`U(\theta)=e^{-i\theta\sigma_z/2},\qquad U(2\pi)=-\mathbb 1,\quad U(4\pi)=\mathbb 1`,
    String.raw`J^\mu=\bar\psi\gamma^\mu\psi,\qquad \{\gamma^\mu,\gamma^\nu\}=2\eta^{\mu\nu}`,
  ],
  gravity: [
    String.raw`\Gamma'^{\mu}_{\alpha\beta}=\frac{\partial x'^\mu}{\partial x^\nu}\frac{\partial x^\rho}{\partial x'^\alpha}\frac{\partial x^\sigma}{\partial x'^\beta}\,\Gamma^\nu_{\rho\sigma}+\frac{\partial x'^\mu}{\partial x^\nu}\frac{\partial^2 x^\nu}{\partial x'^\alpha\,\partial x'^\beta}`,
    String.raw`\frac{D^2\xi^\mu}{d\tau^2}=-R^\mu{}_{\nu\alpha\beta}\,u^\nu\xi^\alpha u^\beta`,
  ],
  connections: [
    String.raw`F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu,\qquad A_\mu\to A_\mu+\partial_\mu\alpha\;\Rightarrow\;F_{\mu\nu}\to F_{\mu\nu}`,
    String.raw`\Delta\varphi=\frac{q}{\hbar}\oint A_\mu\,dx^\mu=\frac{q\,\Phi}{\hbar},\qquad q=-e`,
  ],
  topology: [
    String.raw`C=\frac{1}{2\pi}\int_{\mathrm{BZ}}\Omega(\mathbf k)\,d^2k\;\in\;\mathbb Z`,
    String.raw`\sigma_{xy}=\nu\,\frac{e^2}{h},\qquad \nu\in\mathbb Z`,
  ],
  entanglement: [
    String.raw`|\Phi^+\rangle=\tfrac{1}{\sqrt2}\left(|00\rangle+|11\rangle\right)\neq|a\rangle\otimes|b\rangle`,
    String.raw`|S|=|E(a,b)-E(a,b')+E(a',b)+E(a',b')|\le2\ \text{(local)},\quad\le2\sqrt2\ \text{(quantum)}`,
  ],
  frontiers: [
    String.raw`[x^\mu,x^\nu]=i\theta^{\mu\nu},\qquad \Delta x\,\Delta y\ge\tfrac12|\theta^{xy}|`,
    String.raw`(e_1e_2)e_3=-e_6,\qquad e_1(e_2e_3)=+e_6`,
  ],
  bridge: [String.raw`ab=a\cdot b+a\wedge b`, String.raw`R=e^{-B\theta/2},\qquad v\mapsto Rv\tilde R`],
} as const;

/** Electron around positive enclosed flux; circular gauge for the partial-path readout. */
export function magneticLoop(angleDegrees: number) {
  const fluxFraction = .25;
  const fluxWebers = fluxFraction * 6.62607015e-34 / 1.602176634e-19;
  const phaseDegrees = angleDegrees === 0 ? 0 : -fluxFraction * angleDegrees;
  return {fluxFraction, fluxWebers, phaseDegrees, phaseRadians:phaseDegrees*Math.PI/180};
}

/** GM/r³ at Earth's surface (s⁻²): the tidal gradient that a falling lift cannot remove. */
export const EARTH_TIDE = 3.986004418e14 / 6.371e6 ** 3;

/**
 * A lift in free fall near Earth's surface, `seconds` after the cable snaps (from rest).
 * dropMetres: how far everything has fallen (seen from the ground).
 * squeezeMetres: how far an apple 1 m beside the centre has moved towards it (horizontal tide, both views).
 */
export function fallingLift(seconds: number) {
  const t = Math.max(0, seconds);
  return { dropMetres: 0.5 * 9.81 * t * t, squeezeMetres: 0.5 * EARTH_TIDE * 1 * t * t };
}
