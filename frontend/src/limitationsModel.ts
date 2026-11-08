/** Joint outcomes in one common measurement basis; rows A, columns B. */
export function jointProbabilities(paired: boolean): number[][] {
  return paired ? [[.5,0],[0,.5]] : [[.25,.25],[.25,.25]];
}
/** The right loop cannot cross the excluded disk of radius 28. */
export function loopRadii(percent: number): [number,number] {
  const radius=51*(1-Math.min(100,Math.max(0,percent))/100);
  return [Math.max(0,radius),Math.max(32,radius)];
}
export const LIMITATION_EQUATIONS = {
  scope:String.raw`v_{\mathrm{new}}=(\cos\theta,-\sin\theta)`,
  bridge:String.raw`ab=a\cdot b+a\wedge b`,
};
