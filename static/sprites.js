// AURA pixel assets. Every sprite is a list of 8x8 char-map frames rendered to
// crisp SVG. Silhouettes differ per berry type so the three stay distinguishable
// at cell size and for colour-blind viewers.
//
// Animation is deliberately low-framerate and pixel-snapped. The agent's walk
// cycle is driven by the simulation step counter, not by wall-clock time, so what
// you see moving IS the experiment advancing.

export const PAL = {
  bg0: '#0d0b1f', bg1: '#16142b', cellA: '#221f3d', cellB: '#2a2649',
  line: '#453d6b', dim: '#7a72a8', txt: '#c9c2ee', white: '#f2f0ff',
  red: '#e43b44', redDk: '#8f2027', redLt: '#ff9aa0',
  blue: '#4d9be6', blueDk: '#25507f', blueLt: '#a8ddff',
  yel: '#ffd541', yelDk: '#a67514', yelLt: '#fff0a8',
  green: '#63c74d', cyan: '#c0e8ff', cyanDk: '#5f8fb5',
  orange: '#ff8426', mag: '#d83f87',
};

// '.' = transparent. Every entry is an array of frames.
const MAPS = {
  red: [[
    '...GG...', '..kkkk..', '.kRhRRk.', 'kRhRRRRk',
    'kRRRRRRk', 'kRRRRRRk', '.kRRRRk.', '..kkkk..',
  ], [
    '...GG...', '..kkkk..', '.kRRhRk.', 'kRRRRRRk',
    'kRhRRRRk', 'kRRRRRRk', '.kRRRRk.', '..kkkk..',
  ]],
  blue: [[
    '...k....', '...kk...', '..kBBk..', '.kBhBBk.',
    '.kBBBBk.', 'kBBBBBBk', '.kBBBBk.', '..kkkk..',
  ], [
    '...k....', '...kk...', '..kBBk..', '.kBBhBk.',
    '.kBhBBk.', 'kBBBBBBk', '.kBBBBk.', '..kkkk..',
  ]],
  yellow: [[
    '...kk...', '..kYYk..', '.kYhYYk.', 'kYYYYYYk',
    'kYYYYYYk', '.kYYYYk.', '..kYYk..', '...kk...',
  ], [
    '...kk...', '..kYYk..', '.kYYhYk.', 'kYhYYYYk',
    'kYYYYYYk', '.kYYYYk.', '..kYYk..', '...kk...',
  ]],
  // walk cycle: legs together / mid-stride
  agent: [[
    '...a....', '..CCCC..', '.CCCCcc.', '.CkCCkc.',
    '.CCCCcc.', '..Cccc..', '.cc..cc.', '.c....c.',
  ], [
    '...a....', '..CCCC..', '.CCCCcc.', '.CkCCkc.',
    '.CCCCcc.', '..Cccc..', '..cccc..', 'c......c',
  ]],
  // antenna blink while the LLM is deciding
  agentThink: [[
    '...A....', '..CCCC..', '.CCCCcc.', '.CACCAc.',
    '.CCCCcc.', '..Cccc..', '.cc..cc.', '.c....c.',
  ], [
    '...a....', '..CCCC..', '.CCCCcc.', '.CkCCkc.',
    '.CCCCcc.', '..Cccc..', '.cc..cc.', '.c....c.',
  ]],
  // three-frame one-shot, played on a prediction mismatch
  burst: [[
    '........', '........', '...oo...', '..oOOo..',
    '..oOOo..', '...oo...', '........', '........',
  ], [
    '........', '..o..o..', '.oOOOOo.', '.OOOOOO.',
    '.OOOOOO.', '.oOOOOo.', '..o..o..', '........',
  ], [
    'o......o', '..o..o..', '.o.oo.o.', 'o..oo..o',
    'o..oo..o', '.o.oo.o.', '..o..o..', 'o......o',
  ]],
  skull: [[
    '..kkkk..', '.kwwwwk.', 'kwKwwKwk', 'kwwwwwwk',
    'kwwKKwwk', '.kwwwwk.', '..kwwk..', '..k.k.k.',
  ], [
    '..kkkk..', '.kwwwwk.', 'kwKwwKwk', 'kwwwwwwk',
    'kwwKKwwk', '.kwwwwk.', '..kwwk..', '..kk.kk.',
  ]],
  target: [[
    'oo....oo', 'o......o', '........', '........',
    '........', '........', 'o......o', 'oo....oo',
  ], [
    '........', '.oo..oo.', '.o....o.', '........',
    '........', '.o....o.', '.oo..oo.', '........',
  ]],
};

const INK = {
  red:    { k: PAL.redDk,  R: PAL.red,  h: PAL.redLt,  G: PAL.green },
  blue:   { k: PAL.blueDk, B: PAL.blue, h: PAL.blueLt },
  yellow: { k: PAL.yelDk,  Y: PAL.yel,  h: PAL.yelLt },
  agent:  { a: PAL.yel, A: PAL.orange, C: PAL.cyan, c: PAL.cyanDk, k: PAL.bg0 },
  burst:  { o: PAL.orange, O: PAL.yel },
  skull:  { k: PAL.bg0, w: PAL.white, K: PAL.mag },
  target: { o: PAL.white },
};
INK.agentThink = INK.agent;

export const FRAMES = (name) => MAPS[name].length;

/** Render one frame of an 8x8 sprite to an inline SVG string. */
export function sprite(name, px = 6, frame = 0) {
  const frames = MAPS[name];
  const map = frames[((frame % frames.length) + frames.length) % frames.length];
  const ink = INK[name] || {};
  let rects = '';
  map.forEach((row, y) => {
    [...row].forEach((ch, x) => {
      const fill = ink[ch];
      if (fill) rects += `<rect x="${x}" y="${y}" width="1" height="1" fill="${fill}"/>`;
    });
  });
  return `<svg width="${8 * px}" height="${8 * px}" viewBox="0 0 8 8" `
       + `shape-rendering="crispEdges" aria-hidden="true">${rects}</svg>`;
}
