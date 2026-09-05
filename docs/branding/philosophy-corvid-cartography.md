# Corvid Cartography — libhmm

**The family.** OldCrow's libraries share one visual discipline, established
by the corvus piece: a single corvid held against a field of patient
measurement, drawn as if an ornithologist and a metrologist shared one
drafting table. Ink-dark ground (`#161b23`), marks in the cream of old
survey paper (`#e7e0cf`), and a tarnished gold (`#c8a24e`) spent on no more
than a handful of points, so that when it appears it means something has
been fixed and named. One display serif for one word; a mono whisper for
everything numbered. Each library keeps every rule and changes only three
things: the species, the figure, and what the gold means.

**The species.** The raven — the corvid of hidden things. It is barely
drawn: a form the same black as a wing against a ground almost as dark,
with a few broken contour fragments where the measurement has reached it,
and nothing more. A hidden Markov model is the branding problem stated
plainly: you observe the evidence, never the state.

**The figure.** The evidence is what gets drawn in full. A line of tracks
leads away from the bird and becomes the labeled observation sequence
`y0 … y8`; above it stands the trellis, three states by nine steps, every
transition a hairline. In the corner, the constellation is the
state-transition diagram itself — three stars, directed arcs, self-loops
on the states the path dwells in.

**The gold.** The Viterbi decode — the single path threading the trellis,
the one thing that has been fixed and named — and the raven's eye, the
only point of the hidden thing the instruments ever caught.

**The caption.** `fig. 3 — the decoded path, argmax over all who passed`.
The corvus piece is `fig. 1`; the plates belong to the same imaginary
field guide.
