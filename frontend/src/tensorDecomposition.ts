export type Vec2 = [number, number];
export const EXAMPLE_INPUT: Vec2 = [1, 2];
const A = [[2, 1], [0, 3]];
/** A passive change of orthonormal basis. Dual readers are rows of B inverse. */
export function tensorPieces(degrees: number) {
  const t = degrees * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const basis: Vec2[] = [[c, s], [-s, c]];
  const read = (v: Vec2): Vec2 => [c*v[0]+s*v[1], -s*v[0]+c*v[1]];
  const apply = (v: Vec2): Vec2 => [A[0][0]*v[0]+A[0][1]*v[1], A[1][0]*v[0]+A[1][1]*v[1]];
  const columns = basis.map(e => read(apply(e)));
  const input = read(EXAMPLE_INPUT);
  const terms = [0,1].flatMap(i => [0,1].map(j => {
    const weight = columns[j][i], amount = weight * input[j];
    return {i,j,weight,measurement:input[j],vector:[amount*basis[i][0],amount*basis[i][1]] as Vec2};
  }));
  const result = terms.reduce<Vec2>((a,t)=>[a[0]+t.vector[0],a[1]+t.vector[1]],[0,0]);
  return {basis,input,terms,result};
}
