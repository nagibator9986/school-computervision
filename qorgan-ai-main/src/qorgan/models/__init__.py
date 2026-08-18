"""Model wrappers. The only code in the project that needs a GPU.

Everything that decides anything lives in `qorgan.detection`, which is pure. That split
is what lets the judgement -- the part that determines whether a child gets help -- be
interrogated without a graphics card.
"""
