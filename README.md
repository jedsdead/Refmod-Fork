# MiniMax H3 RefMods for Wan2GP — jedsdead fork

**Version 0.31.0-fork.19** · based on the original port's 0.31.0 · repo: https://github.com/jedsdead/Refmod-Fork

This is a fork of [g3n3rativ3's Wan2GP port](https://github.com/g3n3rativ3/MiniMaxH3Mod-for-WanGP)
of [Luisa's ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).
It fixes several problems that stopped RefMods working reliably in Wan2GP, adds
FL2VA support, and makes a few quality-of-life changes. Mod files are unchanged,
so mods are still interchangeable with the original port and with ComfyUI.

The original port's documentation follows below the line; this section lists
what's different.

## What this fork changes

### Fixes

- **RefMods now reach the prompt.** In the original, a mod's latent went to the
  model with nothing in the prompt tying it to your text, so mods often had
  little visible effect. Each mod now gets its own `<Picture N>`, `<Video N>` or
  `<Audio N>` entry in the prompt, the same as a live reference does.
- **Mods apply on every sliding window**, matching how Wan2GP treats its own
  references. The original injected them into the first window only, so
  anything longer than one window lost them.
- **Your mod selection is no longer lost before generation.** Root cause: Wan2GP
  keeps only the first 5 custom settings a model declares and Ref2VA declares 4,
  so the plugin's second setting was truncated away. Both payloads now share one
  setting. An in-memory fallback remains as a backstop, used only when a task
  genuinely arrives without its payload, and it follows the panel exactly —
  clearing the pickers clears it. Wan2GP can drop
  the plugin's custom setting between the form and the queue. The inline
  panel's selection is now kept and re-applied when that happens, and Wan2GP's
  own form refreshes no longer wipe it. A **Clear armed RefMods** button
  replaces "pick nothing" as the way to turn mods off.
- **Extract no longer renders a video by mistake.** Root cause: Wan2GP keeps
  only the first 5 custom settings a model declares, Ref2VA declares 4, and
  the plugin added 2 — so the extraction key was silently dropped and the
  task ran as a normal render. Both payloads now share one setting, and the
  job is also staged in memory as a backstop.
- **Two-phase generation.** Phase 1 renders at half resolution, and live
  references are shrunk to match; mods weren't. They're now fitted to the
  phase-1 canvas — only ever shrunk, never enlarged — and single-phase runs
  are untouched.
- **Newer Wan2GP's reference-video time budget.** When several reference
  videos together run over Wan2GP's 15-second budget, it trims each one. Mods
  are now trimmed to match instead of crashing the generation.
- **Less repeated work at window boundaries.** Mod files, strength-weighted
  latents and prompt previews are cached instead of being rebuilt every window.

### New

- **FL2VA support**, including FL2VA finetunes such as VDN. In FL2VA the mods
  go straight into the transformer's reference list, the same mechanism the
  ComfyUI original's Apply node uses. Visual mods carry no `<Picture N>` label,
  as in ComfyUI; audio mods are labelled `<Audio N>` (see
  `FL2VA_PROMPT_LABELS`) because a voice reference has little to anchor it
  otherwise. The FL2VA ControlNet model takes mods too (since fork.18); Viggle and TTS models are excluded. The FL2VA
  checkpoint wasn't trained with references, so expect results to vary more
  than in Ref2VA.
- **Mods with both visuals and audio.** A mod can carry a soundtrack as well
  as a look. The audio is injected as its own `<Audio N>` reference, so refer
  to it that way in prompts (`<Video 1>` is the picture; `<Audio 1>` is the
  voice). Set `ATTACHED_AUDIO_AS_SEPARATE_REF = False` to merge it into the
  visual reference as H3's `video_audio` kind instead — that shape means "this
  footage's own soundtrack" rather than a voice to reuse. Attach one at
  extraction time, or from the Library tab's
  **Soundtrack** section for mods you already built. The extra audio counts
  toward the mod's token total (2 tokens per audio latent, ~80 per second).

### Stacked still images: one reference per image

A mod extracted from several stills becomes one image reference *per still*, so
a 3-image mod gets `<Picture 1>`, `<Picture 2>` and `<Picture 3>` and uses three
of the nine image slots. That is the original port's deliberate design, and this
fork keeps it. MiniMax H3's packer gives every "image" reference a single-frame
position grid, so a multi-frame image reference either fails outright or leaves
every frame at the same position, indistinguishable to the model.

ComfyUI behaves differently — there a refmod is one latent block with one tag —
so prompts written for ComfyUI need adjusting. To refer to a stacked mod, name
its pictures together: `<Subject 1> is the man in <Picture 1>, <Picture 2> and
<Picture 3>.` Video mods are already a single `<Video N>`, because the packer's
video branch is built for multi-frame references.

### Quality of life

- Folder management in the Library tab: create folders (`characters/female`
  nests), move any number of selected mods between them, and delete empty ones.
  A move rewrites the mod in place, so the latent, description and any attached
  soundtrack survive, and a name clash is refused rather than overwriting.
- Reference videos have a preview and a trim range in the extractor. With one
  video uploaded it is selected automatically; with several, a picker chooses
  which to preview. A trim limits both the frames encoded and the audio taken
  from that clip, and leaving the range at the full span records no trim.
- Image, video and audio pickers start with one slot each, with buttons to
  add and remove slots, in both the inline panel and the plugin's own tab.
- Delete mods by picking them from a list (several at once if you like), with
  a confirmation tick, instead of typing their paths.
- Every generation logs what happened to your mods, e.g.
  `[H3RefMod] generate(): Ref2VA, window 2, task payload=no, armed=yes`.

### Settings

These live at the top of `patches.py`. The defaults are what this fork ships with.

| Setting | Default | What it does |
|---|---|---|
| `PROMPT_LABELS_FOR_REFMODS` | `True` | Give each mod a `<Picture N>`/`<Video N>`/`<Audio N>` prompt entry. `False` reproduces the original port's behaviour. |
| `INJECT_ON_EVERY_WINDOW` | `True` | Apply mods to every sliding window rather than only the first. |
| `FIT_REFMODS_TO_PHASE_1_CANVAS` | `True` | In two-phase runs, shrink mods to phase 1's half-resolution canvas. |
| `ATTACHED_AUDIO_AS_SEPARATE_REF` | `True` | Inject a mod's soundtrack as its own `<Audio N>` reference. `False` merges it into the visual reference (`video_audio`). |
| `FL2VA_REFMODS` | `True` | Allow mods on FL2VA models. |
| `CONTROLNET_REFMODS` | `True` | Allow mods on the FL2VA ControlNet model (pads its control rows for the mods). |
| `FL2VA_PROMPT_LABELS` | `"audio"` | Which mods get a prompt label in FL2VA. `"audio"` labels audio mods as `<Audio N>`; `"none"` matches the ComfyUI original. |
| `STABLE_REFMOD_LABELS` | `True` | In windows 2+, number mods ahead of the carried-over frame so their `<Picture N>` stays the same as in window 1. |
| `RESCALE_REFMODS_IN_PHASE_2` | `False` | Experimental: rebuild image mods at phase 2's full resolution. |
| `RESCALE_REFMOD_VIDEOS_IN_PHASE_2` | `False` | Same for video mods. Expensive: token cost grows with every latent frame. |
| `REFMOD_RESCALE_MAX_TOKEN_GROWTH` | `2.0` | Cap on how much a phase-2 rebuild may grow a mod's token count. |
| `DISABLE_SOL_WITH_REFMODS` | `False` | Fall back from Sol-Attn to normal attention when mods are in use. |

### Release history

- **0.31.0-fork.19** — **Mod size controls.**
  - **Max size, per mod.** Each picture and video row in the inline panel has
    a *Max size* slider under *Strength*, shown once a mod is picked. It
    starts at the mod's own size (full) and goes down in 32-pixel steps on
    the short edge, keeping the mod's shape. A smaller size means fewer
    tokens and a faster generation, at the cost of fine detail, faces first.
    The readout shows each shrink and roughly what share of the mod's tokens
    is left. It applies to single-phase runs and to phase 2. Audio mods and
    mods already too small to shrink have no slider.
  - **Mods in phase 2** (inline panel, two-phase runs only): *Full* (default)
    keeps the mods as set; *Fit to tile* shrinks each mod to one tile of a
    tiled phase 2, so the four tiles stop paying full price for every mod
    (with tiling off it behaves like Full); *Off* (experimental) runs phase 2
    without the visual mods -- fastest, but faces may drift. Audio mods stay
    in, and in Ref2VA the mods keep their prompt labels. Ref2VA keeps the mods
    in phase 2 anyway when the generation has reference images or videos of
    its own, because it lines phase 2's references up with phase 1's by
    position.
  - **FL2VA phase 1 now shrinks mods to the half-size phase-1 picture**, as
    Ref2VA has done since fork.9. Before, FL2VA mods went into phase 1 at
    full size.
  - Shrinking is done inside the generation with its own, already-loaded
    video VAE: each mod is decoded, resized and re-encoded once, then kept in
    memory for later windows and runs at the same size. When several limits
    apply (Max size, Fit mods to the output size, the phase-1 canvas, a
    phase-2 tile), the smallest wins. Mods are only ever shrunk, never
    enlarged.
- **0.31.0-fork.18** —
  - **RefMods on the FL2VA ControlNet model.** The MiniMax H3 FL2VA
    ControlNet-Union model now takes mods the same way FL2VA does, and the
    inline panel shows for it. ControlNet builds its control rows with blank
    padding for every reference row ahead of the video, before the mods are
    added, so the plugin pads for the mods too -- without that the model
    stops with "control rows must match the packed video rows". The control
    branch only acts on the prompt and the video being made, never on the
    mods. Neither ControlNet nor FL2VA was trained with references, so
    results need testing; lowering *Control Strength* gives the mods more say.
    Write prompts as for RefMod in FL2VA.
  - **Fit mods to the output size** (inline panel, under the text encoder
    options; **off by default**). Ticked, any picture or video mod bigger than
    the video being made is rebuilt at the output's pixel area, keeping its
    own shape. Fewer reference tokens, so generation is faster -- a 704p video
    mod in a 480p generation costs about half as many tokens -- but a shrunk
    mod carries less fine detail, faces first. Mods already that size or
    smaller are left alone. Works in Ref2VA, FL2VA and FL2VA ControlNet. In a
    two-phase Ref2VA run, phase 1 already fits mods to its half-size canvas;
    this also fits them to the full output in phase 2.
- **0.31.0-fork.17** — **Decompile tab.** A fourth tab rebuilds what a mod
  holds as ordinary files: one PNG per picture, an MP4 for a video mod (with
  its soundtrack), and a WAV for any audio. It also lists what the mod was
  made from, its mode, size, tokens and soundtrack length, and shows its
  stored encoder frames instantly, without decoding. The files are
  reconstructions of the stored latents, not the original sources: what
  extraction discarded (resolution, the parts outside a trim, pooled detail
  in training mode, the pictures behind a merge) can't come back. Decoding
  runs as a short task; the files go to
  `loras/refmods_plugin/decompiled/<mod name>/`.
- **0.31.0-fork.16** — **Soundtrack length, and a 2-second minimum.**
  - A new *Soundtrack length* slider in the Extract tab's *Soundtrack*
    section (2-15s, **4s by default**) sets how much audio a mod keeps -- an
    audio-only mod, an audio file attached to pictures or a video, or a
    clip's own audio -- independent of the video length. *Reference duration
    to use* is now for video only (it still defaults to ~2.5s).
  - **Audio files shorter than 2s are refused** (H3's documented minimum for
    an audio reference), before anything runs, with the reason shown in the
    status box. The one exception is *Use the clip's own audio*: the clip can
    still make a good visual mod, so its audio is kept even when under 2s,
    with a warning in the status box that the voice may not be picked up.
    The Library's *Soundtrack* tool refuses short clips, and its *Seconds to
    use* now starts at 2s.
  - The audio-file warning now compares against *Soundtrack length*, and
    flags a file under 2s up front.
- **0.31.0-fork.15** — **An empty RefMod panel now always means no mods.**
  The panel's selection is also kept as a server-wide fallback, for a task
  that arrives without its own copy. A page reload (or a rebuilt panel) shows
  empty slots without any slot firing a change, so that fallback kept the
  last selection and quietly injected mods the panel no longer showed. Now,
  every time Wan2GP saves the form or builds a task for a RefMod-capable
  model, the fallback is reset to what the panel actually holds, and the
  console says so when it clears.
- **0.31.0-fork.14** — **Merge into one picture** (Extract tab, *Settings*,
  training mode), ported from the ComfyUI RefMod pack's *merge* option.
  Instead of keeping every picture, several sources become ONE small picture
  of what they have in common, at one reference's token cost however many
  go in. It starts from the average of the sources' pooled grids and, with
  *Identity refinement steps* above 0, refines that grid to match every
  full-resolution source at once. What agrees across the sources at the same
  place in the frame survives; backgrounds and unique details are averaged
  away, so it works best on similarly framed shots and for a general look
  rather than a precise face. Encode mode always stacks. Merged mods are
  tagged `merged N refs`.
- **Fix (fork.14): soundtracks on picture mods were cut to half a second.**
  A voice attached to a single picture, a picture stack or a merged mod was
  sized from the pictures' latent frames, which gave it ~0.5s -- too short
  for H3 to use. Picture mods now get *Reference duration to use* worth of
  audio (pictures taken from a video: their span), video mods still match
  their clip, and no soundtrack is shorter than 2s, H3's documented minimum
  for an audio reference. Picture mods extracted with a soundtrack before
  fork.14 should be re-extracted, or have their soundtrack re-added from the
  Library (*Soundtrack* section).
- **0.31.0-fork.13** —
  - **Soundtracks count in the readout.** A soundtrack attached to an image or
    video mod is injected as its own audio reference, and the panel's *Audio*
    total now includes it ("from 2 mod soundtracks"), along with its
    `<Audio N>` label.
  - **Include audio, per mod.** Image and video slots holding a mod with a
    soundtrack get an *Include audio* box (ticked by default). Untick it to
    use the mod's look without its voice; the readout and labels follow.
  - The per-slot boxes now start fresh whenever a different mod is picked in
    that slot.
- **0.31.0-fork.12** — quality-of-life updates to the RefMod panel:
  - **No more RefMods text box** on Wan2GP's Media Generator form. The
    setting still exists (the selection travels with every task, queue entry
    and saved settings file); it's just hidden, and it no longer clutters a
    finished video's settings view either.
  - **"Send as one video" is now per mod.** Instead of one panel-wide
    checkbox, each image slot gets a small *Send as one video* box, shown only
    when the slot holds a mod made from several pictures. Selections saved
    before fork.12 keep their old panel-wide choice.
  - **One slot per kind to start with**, plus *Add* and *Remove last* buttons
    for image, video and audio mods, instead of every slot shown at once.
  - Picture stacks saved as "video" (older versions of this plugin, and
    ComfyUI) now appear in the *image* pickers, where they're injected as
    pictures and can be sent as one video.
  - *Clear armed RefMods* now also empties the slots, so the panel never
    shows mods that aren't armed.
- **0.31.0-fork.11** — in the RefMod panel (inline and Generate tab), the
  multi-picture checkbox, the reference/label readout and the *Text encoder
  options* (now open by default) are back at the top, above the mod pickers,
  so they're hard to miss. Also fixes the Extract tab's audio-length warning,
  which pointed "above" to *Reference duration to use* after it moved into
  *Settings*.
- **0.31.0-fork.10** — **Turn reference videos into a picture stack** (Extract
  tab): instead of a clip, a video becomes still pictures, taken at a chosen
  number per second (default 1) over the same span a clip would use (the trim,
  up to *Reference duration to use*). The sharpest frame of each slot is kept,
  skipping motion blur and blinks. Each picture costs about as much as 3-4
  frames of a clip, so a stack keeps identity at a fraction of the tokens, but
  not motion. With *Use the clip's own audio*, the voice covers the whole span.
  A live readout shows the picture count and estimated tokens, and warns past
  9 pictures (H3's documented limit for picture references).
- **Tidier UI** (fork.10):
  - *Extract*: name and mode first, with description and concept type in a
    collapsed accordion; image and video sources side by side, each with its
    own options beneath (background removal under images, picture stack under
    videos); the soundtrack controls together, with the long explanation
    collapsed; one *Settings* group, where the training-mode sliders only
    appear in training mode; *Save* and *Store encoder frames* next to the
    Extract button.
  - *Library*: the table on top, and every tool (prompt hint, rename, folders,
    soundtrack, encoder frames, maintenance, delete) in its own collapsed
    accordion.
  - *Mod pickers*: the multi-picture checkbox sits with the label readout it
    changes, below the pickers; the encoder frame counts are in a collapsed
    *Text encoder options* accordion (moved back to the top in fork.11).
  - "Prompt encoder" is now called the *text encoder* throughout.
- **0.31.0-fork.9** — restores fork.7's two-phase behaviour: mods are again
  shrunk to phase 1's half-resolution canvas
  (`FIT_REFMODS_TO_PHASE_1_CANVAS` is back to `True`), matching how Wan2GP
  sizes its own references. Single-phase runs are unaffected.
- **0.31.0-fork.8** — ideas from ComfyUI-Fantastic-MiniMaxH3-PromptBuilder:
  - **Encoder frames.** New mods store the pictures the text encoder is
    shown (JPEG, inside the mod file), so generation no longer decodes them.
    Older mods still work, decoded once per session as before. Library →
    *Encoder frames* stores them for existing mods (selected, or every mod
    missing them), and the table has an *encoder frames* column. Moving,
    renaming and attaching a soundtrack keep them.
  - **Multi-picture mods as one video reference.** A checkbox in the RefMod
    panel sends a mod made from several pictures as one `<Video N>` instead of
    one `<Picture N>` per picture, with *all* or *up to N* pictures shown to
    the text encoder (every picture always reaches the model). Works with
    existing mods. The panel now lists the label each mod will get.
  - Mods used below strength 1 are softened for the text encoder too, so
    it isn't shown a stronger reference than the model gets.
  - `STABLE_REFMOD_LABELS` now defaults to `True`: a mod keeps the same
    `<Picture N>` in every sliding window.
  - ComfyUI picture stacks (saved there as video mods) are recognised as
    pictures, and metadata from other tools is kept when a mod is rewritten.
  - Two-phase runs no longer shrank mods in phase 1 (reverted in fork.9).
  - Fixed the audio-vs-visual pre-flight check reading the wrong setting key.
  - **Video mods: frames shown to the text encoder** in the RefMod panel:
    *up to N* (default 8, as before) or *all*. Fewer frames make the prompt
    encode lighter; the video mod itself always reaches the model in full.
  - **Store encoder frames** option when extracting (on by default).
  - The Library's Refresh now also updates the *Mods to move* list and the
    folder pickers, so a newly extracted mod can be moved straight away.

- **0.31.0-fork.7** — fixes `"Value: X is not in the list of choices"` after
  moving, renaming or deleting a mod that was selected in the Library tab: the
  pickers kept a value that no longer existed. Every picker now drops a stale
  selection and keeps one that survived.
- **0.31.0-fork.6** — folders can be created, filled and removed from the
  Library tab: make a folder (nested with `/`), select any number of mods and
  move them into it or back to the top level, and delete folders once empty.
- **0.31.0-fork.5** — `requirements.txt` no longer asks for
  `opencv-python-headless`. Wan2GP already requires `opencv-python`, and the two
  are the same library packaged differently and not meant to share an
  environment; on some installs pip failed on it and took the whole plugin
  install down ("Failed to install dependencies"). Nothing else was in the file,
  so the dependency step is now a no-op.
- **0.31.0-fork.4** — reference videos can be previewed and trimmed in the
  extractor: pick an uploaded clip to watch it, and set the span the mod is
  built from. Trims are per video, and also limit the soundtrack taken by
  *Use the clip's own audio*.
- **0.31.0-fork.3** — **Use the clip's own audio** when extracting a video mod:
  one clip of someone talking gives the mod both the look and the voice. The
  Library's Soundtrack section now accepts a video file too, so an existing mod
  can be given the voice from the clip it was built from — if you still have
  that file, since a mod records no source paths.
- **0.31.0-fork.2** — rebased onto the original port's 0.31.0. Two things this
  fork had added independently are now upstream and the upstream versions are
  used instead: the third reference-video/audio slots (detected from
  `generate()`'s signature) and the reference-video budget trim on a video mod.
  The trim is extended to carry this fork's extra fields, so a trimmed mod keeps
  its attached soundtrack. Stacked still images keep upstream's behaviour of one
  image reference per still (see below).

Releases below were based on the original port's 0.30.2.

- **0.30.2-fork.1.8** — the extractor form accepts audio together with image or
  video sources. The backend had supported it since fork.1.3, but a leftover
  check in the form still refused the combination before submitting.
- **0.30.2-fork.1.7** — clearing the pickers clears the selection again. It used
  to be sticky (from when the task payload was being dropped), so an empty panel
  kept injecting the last selection into every generation — taking reference
  slots from the generation's own references and making them appear broken.
- **0.30.2-fork.1.6** — the injection log now reports which reference slots
  were filled and the final video/audio prompt-type flags, so a reference
  that reached a slot can be told apart from one that didn't.
- **0.30.2-fork.1.5** — fixes a crash when injecting a mod that carries a
  soundtrack (`_RefModAudioSentinel() takes 2 positional arguments but 3 were
  given`).
- **0.30.2-fork.1.4** — a mod's attached soundtrack is injected as its own
  `<Audio N>` reference (the trained voice-reuse path in Ref2VA) rather than
  merged into the visual reference as H3's `video_audio` kind, which says the
  sound belongs to that footage. Refer to it in prompts as `<Audio N>`.
- **0.30.2-fork.1.3** — a mod can carry a soundtrack as well as a look:
  extraction accepts audio alongside image/video sources, and the Library
  tab can add, replace or remove a soundtrack on mods you already have.
- **0.30.2-fork.1.2** — root-cause fix for the plugin's settings being
  dropped: both payloads now travel in a single `h3_refmod` custom setting.
  Wan2GP keeps only the first 5 settings a model declares and Ref2VA already
  declares 4, so the plugin's second setting was silently truncated — which
  is why extraction could run as an ordinary render.
- **0.30.2-fork.1.1** — audio mods are labelled `<Audio N>` in FL2VA, so the
  model is told what the reference audio is for; the staging log reports audio
  row counts.
- **0.30.2-fork.1** — first fork release: the fixes and quality-of-life changes
  above, FL2VA support, and three reference-video/audio slots.

### Versioning

Fork versions take the form `<original port version>-fork.<fork release>`, so
`0.30.2-fork.1` is this fork's first release built on the original's 0.30.2.
Wan2GP's plugin manager sorts it above 0.30.2 and below the next original
release, and compares fork releases numerically, so `fork.10` is newer than
`fork.9`.

---


# MiniMax H3 RefMods -- a Wan2GP plugin

No-training "reference mods" for MiniMax H3 **Ref2VA**: compress an image or
video reference into a small `.safetensors` file once, then reuse it in later
generations at any strength -- without re-encoding the original picture/clip
every time, and without needing to keep the original file around at all.

This is a port of the idea and math from
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)
(MIT License, (c) 2026 Luisa/luisacaotica) onto [Wan2GP](https://github.com/deepbeepmeep/Wan2GP),
which already ships its own native MiniMax H3 implementation using the exact
same 24-channel video VAE. `core.py` in this plugin is close to a direct port
of that project's `core.py` (which has no ComfyUI-specific code at all, so it
travels almost unchanged); everything else (storage, the UI, and the
generation hooks) is new, written specifically for Wan2GP's plugin API and
its own `MiniMaxH3Pipeline`.

**Mods made by either tool are interchangeable** -- they're the same VAE
latent in the same `.safetensors` layout, so a mod extracted in ComfyUI can be
dropped into `loras/refmods_plugin/minimax_h3/` here and used directly, and vice versa.

## Two ways to use RefMods

**Recommended: the inline panel on Wan2GP's own Media Generator page.** Once
a MiniMax H3 Ref2VA model is selected there, an extra **"MiniMax H3 RefMods
(inline)"** accordion appears (near the LoRAs Multipliers field). Pick up to
9 image + 2 video mods with a strength each and hit *that page's own*
Generate button as usual -- every native field (resolution category/budget,
frame count and its duration in seconds, steps, sampler, the full
attention-mode list, memory profile, FPS override, output filename
template, text encoder/VAE variant, DiT priority, LoRAs, sliding window --
everything) works completely unchanged, because none of it is duplicated:
this plugin only adds the RefMod picker itself, on the same page. See "How
the inline panel works" below for the mechanism and its one caveat.

**Alternative: this plugin's own Extract / Library / Generate tab.** Useful
if you'd rather not touch the main form, or want a second, independent
generation queue. Its Generate section covers the fields people tune most
often directly, plus a **Sync from the main form** button that copies every
other current setting from the Media Generator tab in one click.

Either way, extraction always happens from this plugin's own **Extract**
tab.

## Why a reference mod at all

MiniMax H3's reference path (`Ref2VA`) works by VAE-encoding your reference
image/video, patchifying it, and letting every block of the transformer
attend to those tokens. A full-resolution video reference is expensive: it
can contribute thousands of tokens *every single generation*. A RefMod is the
same reference, saved once:

- **`training` mode (default)**: the encode is average-pooled down to a tiny
  grid (e.g. 16x16 = 64 tokens/frame) and optionally refined with a few
  gradient steps against the full encode (a handful of seconds, no diffusion
  model involved -- this is the only "training" happening anywhere). Cheap,
  carries concept/style/motion well, softer on fine identity at small grids.
- **`encode` mode**: the full-resolution VAE encode is kept as-is (same cost
  as a live reference, but you only pay the encode once and can reuse it
  forever). Best for identity/character fidelity.

At generation time the saved latent is handed back into the exact same
reference-conditioning path a live image/video reference would use, so it
goes through the same attention machinery -- only the token count changes.

## Installation

1. Copy this whole folder into your Wan2GP `plugins/` directory, e.g.:
   `plugins/wan2gp-minimax-h3-refmod/`
2. Start Wan2GP, open the **Plugins** tab, enable **MiniMax H3 RefMods**,
   save settings, and restart Wan2GP (standard Wan2GP plugin flow).
3. A new **MiniMax H3 RefMods** tab appears with three sections: **Extract**,
   **Library**, and **Generate**.

No extra Python dependencies beyond what Wan2GP already ships (torch,
safetensors, numpy, Pillow). `requirements.txt` only lists an optional video
decoding backend as a nice-to-have.

The plugin stores the current selection in one custom setting on MiniMax H3's
model definition, so it travels with every task, queue entry and saved
settings file. Since fork.12 that setting is hidden on Wan2GP's Media
Generator form -- there is no text box to leave blank any more.

## Using it

Before extracting or generating, you can click **Check setup for this
model** (next to the model dropdown, in this plugin's own tab) to confirm
the plugin's hooks are actually active for the model you selected -- it
reports whether the pipeline patch loaded correctly and whether that model
declares the two `custom_settings` ids RefMods rely on. If either check
fails, extraction/generation will silently fall back to a normal render
instead of doing what you asked, with nothing else in the log to explain why
-- so this is worth a quick check the first time, or after updating Wan2GP.

### The inline panel (Media Generator page)

Select a MiniMax H3 Ref2VA model, then open the **"MiniMax H3 RefMods
(inline)"** accordion. Pick mods, set strengths/copies, optionally a master
retention and a curve, then just use the page's own Generate button --
nothing else to configure, nothing else changes. Selections here apply
immediately and don't require a separate "Apply" step, and don't affect
anything if a non-MiniMax-H3 model is selected instead.

### Extract

Upload one or more reference images and/or up to two reference videos (the
same "Use Two Reference Videos" input MiniMax H3 Ref2VA natively supports),
name the mod, pick `training` or `encode` mode, and hit **Extract & Save
RefMod**. Every reference you provide -- images and video(s) alike -- gets
encoded and stacked into this one mod file. This briefly runs as a real
(near-instant) generation task on the MiniMax H3 Ref2VA model you selected,
so it can reuse that model's already-loaded VAE instead of this plugin
trying to load weights on its own. **You will see the normal generation
progress bar for a few seconds and then no video will appear -- that's
expected.** The mod was still saved; check the status box under the button,
or the Library tab, to confirm.

**Mode is the single most consequential choice here** -- it controls what
actually ends up in the saved tensor, unlike the metadata-only fields below:
- **`encode`**: the full VAE-encoded latent is kept as-is, at full fidelity
  -- identical to what a live reference would produce. Best for a person's
  precise face/identity. Bigger file, more tokens spent at generation time
  (same cost as a live reference), and the "Identity refinement steps"
  option is hidden in encode mode -- there's nothing left to refine.
- **`training`**: the latent is average-pooled down to a tiny grid (the
  "Pool grid" sliders under Settings) and optionally refined with a few hundred
  gradient steps against the full encode ("Identity refinement steps",
  Advanced -- optimizing only the small saved latent's own values, no model
  weights involved). Pooling is genuinely lossy: fine detail (eyes, nose,
  mouth proportions) gets averaged away, which is fine for a general
  concept/style/pose but is exactly what causes a specific face to drift
  toward a generic "chubbier/older" look -- hence the live warning under the
  pool grid sliders when `concept_type=identity` is combined with a small
  grid. Smaller file, fewer tokens at generation time.

**Reference duration to use (seconds) -- video AND audio** shows and sets
this directly in real seconds of the source (0.1s steps) rather than the
abstract "latent frames" count the underlying extraction actually works in
-- so if your video (or audio) is 4 seconds long, you can just drag the
slider to 4.0s instead of guessing what frame-count number that
corresponds to. **This one slider bounds both video-kind and audio-kind
extraction** -- there's no separate audio duration control, and it defaults
to a short, video-sized value (2.5s), so double-check it before extracting
audio: a live warning appears next to the audio upload field if the
uploaded file is longer than the current setting, since leaving it at
default silently keeps only the first few seconds of a longer recording
(functionally fine if just the voice's timbre matters, since even a couple
of seconds usually carries that, but not what you asked for if you wanted
the whole clip). It goes up to 14.9s -- MiniMax H3 Ref2VA's own 15-second
reference cap works out to about 90 latent frames once the video VAE's
causal 4:1 temporal compression is accounted for (verified: 90 latent
frames ≈ 357 pixel frames ≈ 14.9s at 24fps; the next notch up would already
be over 15s) -- and audio has its own, separately-verified 15s cap too (see
"Audio RefMods" below). This is a **shared** budget within each kind: if a
generation combines two video-kind mods (or two audio-kind mods), their
durations *add together* against that kind's own 15s ceiling, so a single
mod extracted near 14.9s leaves no room for a second one of the same kind
alongside it -- the live counter (Generate tab / inline panel) always shows
the real total for whatever's currently selected, per kind.

A source video always contributes a **contiguous** clip from its start,
matching the slider as closely as the video VAE's own frame grid allows --
in both modes. (An earlier version of this plugin sampled `encode` mode's
frames sparsely across the *entire* source video instead of a contiguous
prefix; the video VAE has no idea a "frame" was pulled from 8 seconds in
rather than 0.3 seconds in, so it compressed the scattered sample as if it
were a real, short, sequential clip -- silently breaking the slider's
promise. Fixed.) Audio extraction takes a contiguous clip from its start
the same way.

Two things can still make the *saved* mod end up shorter than what you
asked for -- extraction always tells you plainly when this happens, with
the requested vs. actual duration:
- In `training` mode, the slider is a **ceiling, not a guarantee**: pooling
  can't invent seconds the source video didn't produce enough of once
  VAE-compressed, so a short source video ends up shorter than requested no
  matter how high the slider is set. (Doesn't apply to audio -- there's no
  `training` mode for it.)
- **`Max tokens`** (Advanced, defaults to 65536 -- unlocked, so it stays out
  of the way for most extractions) applies *after* encoding, and `encode`
  mode at a high "Ref resolution" can still burn through even that: a
  1024px `encode`-mode video ref costs 1024 tokens per frame, so 65536
  tokens fits about 64 frames (~10.6s). Past that point -- or with a lower
  budget set deliberately -- the mod still gets saved, just shorter. Audio
  costs far fewer tokens per frame (2, stereo) so this is rarely the
  binding constraint for it -- the duration slider itself usually is.

**Automatic background removal (optional).** Same "Automatic Removal of
Background behind People or Objects in Reference Images" toggle Wan2GP's own
Media Generator form offers for reference images -- same `rembg` call and
alpha-matting parameters, so a mod extracted with this on looks consistent
with a live reference image processed the same way. Defaults to **Keep
Backgrounds behind all Reference Images**. Only applies to the reference
image(s), not to reference videos, matching Wan2GP's own behavior. Applied
in pixel space before encoding, so it works the same in both `training` and
`encode` mode.

**`keyword - description` and `Concept type`** (in the *Description and concept
type* accordion under the mod name) are the
opposite of Mode: **pure metadata with no effect on generation whatsoever**.
Nothing in a mod file is ever read by the model automatically -- MiniMax
H3's Ref2VA path has no image-embedding / CLIP-Vision-style channel to hang
an automatic clue off of, so text typed into your prompt is the only
channel that actually reaches the model. They only become useful through
the Library tab's **Prompt hint** section (or by copying them into your
prompt yourself):
- `keyword - description`, written the same way you'd write a LoRA training
  caption, is the text that gets used.
- `Concept type` just prefixes it in that output (e.g.
  `identity: ginger woman, tattooed neck`), the same way the original
  ComfyUI pack's loader does -- and also drives the live pool-grid warning
  mentioned above.

### Audio RefMods

MiniMax H3 Ref2VA also supports up to 2 **direct** audio references ("Use
one/two audio references", flags `A`/`B`) -- distinct from "Use
reference-video soundtrack(s)" (flag `K`), which extracts audio *from* a
reference video rather than taking a standalone audio file. This plugin
covers the direct case with its own mod kind.

**Extracting one.** Upload a file in the **Reference audio** field on the
Extract tab -- it can't be combined with image/video sources in the same
mod (an audio VAE latent `[1, 32, 2, T]`, channels x stereo x time, is
structurally incompatible to stack alongside a visual one
`[1, 24, T, H, W]`); providing both is refused with a clear error rather
than silently mixed or dropped. Audio mods are always extracted at full
VAE fidelity -- there's no spatial grid to pool the way image/video mods
can in `training` mode, so `Mode` above the upload fields doesn't apply to
audio at all. There's no separate audio duration control -- see "Reference
duration to use" above; a live warning appears next to the upload field if
the file is longer than the current setting, since that setting defaults
to a short, video-sized value (2.5s). `Max tokens` and `Repeat multiplier`
apply as for image/video, just with the audio VAE's own, exact conversion
rate (40 latents/second at 32kHz, verified against
`components/audio_autoencoder.py`'s own docstring -- not an fps-based
estimate the way the video duration figure is).

**Using one.** Same as image/video RefMods: pick it in an "Audio Mod 1/2"
slot (Generate tab or the inline panel) and it goes straight into
`audio_guide`/`audio_guide2`, one mod per native slot, the same two slots
"Use one/two audio references" itself uses -- so at generation time it's
indistinguishable from having recorded that exact audio and uploaded it
live. If the current generation already has "Use reference-video
soundtrack(s)" turned on, audio-kind RefMods are skipped with a clear log
message rather than silently overwritten -- the two features read the same
two underlying slots for genuinely different purposes and can't share them
in one generation.

The live reference-budget counter tracks audio the same way it tracks
images and video, as a third, **separate** 15-second budget (video and
audio each have their own native cap -- they don't share one).

### Library

Lists every saved mod (name, kind, mode, token count, file size, description)
read straight from disk. Delete mods you no longer need. Mods live in
`loras/refmods_plugin/minimax_h3/` at the root of your Wan2GP install.

**Subfolders.** You can organise mods into any folder structure you like
inside that directory -- `characters/`, `characters/voices/`, `styles/`,
however deep you want. Every mod picker (this tab, the plugin's own
Generate tab, and the inline panel on the Media Generator page) has a
**Folder** dropdown above the rows listing every subfolder found, plus
`(all folders)` to see everything at once and `(main folder only)` to see
just the top level -- picking `(all folders)` is how you get back out of a
subfolder. Mods inside a subfolder are named by their relative path
(`characters/tanya`), which is what you'd type into the Delete field or use
anywhere else a mod name is expected.

**Mods from different folders can be combined freely in one generation.**
The Folder dropdown only changes what the pickers *offer* -- anything
already selected stays selected and stays listed, even after you browse
somewhere else. So the normal workflow works: browse to `characters`, pick
a face, browse to `voices`, pick a voice, and both are still armed when you
hit Generate.

To put a mod in a subfolder, either drop the `.safetensors` file there
yourself, type a path into the Extract tab's **Mod name** field
(`characters/tanya` -- the folder is created automatically), or use the
rename tool below to move an existing one. Path components are sanitized
the same way plain names always were, and `..` segments are dropped, so a
mod can never be written outside the RefMods directory.

**Fix classification.** Mods extracted purely from several still images (no
video source) were, before this fix, wrongly saved as `video` kind whenever
more than one image was stacked together into the same mod -- because the
underlying latent ends up with more than one frame either way, and the
original logic used "more than one frame" as its only signal instead of
checking whether any of the sources was an actual video. A `video`-kind mod
gets injected as *one* temporal/motion reference (through one of only 2
video slots), while several still images should really be *several*
independent identity references (through the 9-image path) -- so a
misclassified mod could both waste a scarce video slot and get
misinterpreted as a mini "animation" between unrelated photos instead of
several separate looks at the same subject. Click **Fix classification** to
rescan every saved mod's own extraction record and correct this in place --
only the `kind` label changes, the latent data itself is never touched, and
it's safe to run repeatedly (already-correct mods are left alone). New
extractions made with this version already classify correctly from the
start.

**Rename, move, or edit a mod's description.** Pick a mod from the dropdown,
click **Load** to pull its current path/description into the two text boxes
below, edit either one, then **Save changes**. Renaming re-saves the file
under the new name and removes the old one (refused if a mod with that name
already exists, so you never lose one by accident) -- the latent data is
byte-for-byte untouched either way. Typing a plain name keeps the mod in
whatever folder it's currently in (rather than yanking it back to the main
folder, which is almost never what editing a name means); typing a path
(`characters/tanya`) moves it there, creating the folder if needed; and a
leading `/` moves it back to the main folder. Both fields are plain editable
text boxes, so clicking into one and selecting the text (double/triple-click,
or Ctrl+A) then Ctrl+C copies it like any other text on the page -- no
separate copy button needed.

### Generate (this plugin's own tab)

**A large banner at the top of this tab points you to the Media Generator
tab's inline panel instead** -- see "Two ways to use RefMods" near the top
of this document. This tab is kept as a simpler, self-contained fallback.

A generation form covering the fields people tune most often: prompt,
resolution, number of frames, seed, number of inference steps, flow shift,
sampler, reference-image pixel budget, and up to 9 image-kind + 2 video-kind
RefMod slots (mod +
strength + copies), plus a master **retention** dial and an optional
strength **curve** under a small accordion. An **Advanced Mode** accordion
adds:

- **LoRAs** -- pick from the LoRAs installed for the selected model, plus a
  multipliers text field (same syntax as the main form).
- **Steps Skipping** -- Spectrum Feature Forecasting / First Block Cache,
  with the threshold and starting-percentage controls.
- **Sliding Window** -- window size / overlap, used automatically once the
  frame count needs more than one window.
- **Attention** -- override attention backend (e.g. Sol-Attn) and its Start
  Tau sparsity dial.

Click **⟲ Sync from the main form** first to copy every other current
setting from Wan2GP's own Media Generator tab (for whichever model is
selected there) as your starting point -- including anything this panel
doesn't have a dedicated widget for (audio references, live reference
images/video, output filename template, memory profile, text encoder/VAE
variant, category/resolution budget as a single string, etc.). Whatever you
don't sync or override here still gets a valid value from that model's own
factory defaults before submission, so nothing is ever left unset -- though
if you want every native field editable as its own widget rather than
inherited from a snapshot, the inline panel above is the better fit.

## How the inline panel works (and its one caveat)

Wan2GP's own "Custom Settings" fields (the channel this plugin uses to carry
a RefMod selection to the pipeline, see below) are auto-built from a model's
definition but never given a stable `elem_id`, so a plugin cannot bind its
own rich widgets to them directly -- there is no way to make the inline
panel's mod pickers *be* one of those fields. Instead:

- The panel is injected via `insert_after("loras_multipliers", ...)` -- the
  only elem_id Wan2GP's plugin docs guarantee is present on (almost) every
  model's form, so the panel can be added once, at UI-build time, without
  needing per-model wiring.
- Visibility is handled separately, by binding to a hidden text component
  Wan2GP itself already updates on every model switch. Note that Wan2GP
  hands plugins their requested components out of `generate_media_tab`'s
  own `locals()`, so the key is the Python **variable** name
  (`model_choice_target`) -- *not* its elem_id
  (`wangp_model_choice_target`), which is what an earlier version of this
  plugin wrongly requested, silently leaving the panel always-visible. Both
  names are requested now, for compatibility across builds. Its value
  (`"{model_type}|{timestamp}"`) is parsed the same way Wan2GP's own
  `_model_choice_target_model_type` does, and the panel is shown only when
  the model type starts with `minimax_h3_ref2va`. If neither name resolves,
  the panel falls back to always-visible rather than silently disappearing
  -- it still has no effect on any other model either way, this only
  changes whether it's shown.
- Wan2GP auto-saves the whole form continuously as you edit any field (via
  `save_inputs`/`prepare_inputs_dict`), which is also what the real Generate
  button ultimately reads from. So instead of writing into that
  continuously-rebuilt settings dict directly (which the very next field
  edit would silently wipe, since none of the panel's widgets are native
  form fields Wan2GP's own save logic knows about), the panel writes its
  JSON payload into a namespaced key on the session `state` dict this
  plugin owns (`state["_h3refmod_selection"]`) that nothing else in Wan2GP
  ever touches.
- `prepare_inputs_dict` itself is wrapped (via `self.set_global`, Wan2GP's
  own supported way for a plugin to replace one of its globals) so that,
  *every* time it runs -- including the one that matters, right when
  Generate is clicked -- it re-reads that stashed key and folds it into that
  call's `custom_settings`. This survives the autosave cycle by construction,
  since it re-injects on every single call rather than being overwritten by
  one. The wrapper calls the original function unchanged first and only adds
  to its result for MiniMax H3 Ref2VA model types with a non-empty
  selection stashed -- every other model, and MiniMax H3 with nothing
  selected, behave exactly as before.

**The caveat**: this patches a function used by every model in Wan2GP, not
just MiniMax H3 -- a much larger blast radius than this plugin's other
patches, even though the added logic is narrowly scoped and wrapped in its
own `try/except` (a failure here logs a warning and leaves the original
result untouched, it never raises). If you notice *anything* unusual with
non-MiniMax-H3 generations after enabling this plugin, that's the first
place to look, and disabling the plugin removes the patch entirely (nothing
is written to disk). The plugin's own Extract/Library/Generate tab does not
depend on this patch at all and is unaffected either way.

## How the RefMod injection itself works (for anyone auditing this)

Wan2GP's `MiniMaxH3Pipeline.generate()` builds a `refs` list (kind/shape
metadata) and a `visual_latents` list (the actual VAE latents) from whatever
live references you pass it, via two internal helpers,
`_add_image_reference` and `_add_video_reference`. Those lists are local to
`generate()`, so a plugin outside Wan2GP's own source tree cannot reach into
them directly -- and reimplementing that ~300-line method here would be
fragile and guaranteed to drift out of sync with upstream.

Instead, `patches.py` applies four small, targeted monkeypatches to the
already-imported `MiniMaxH3Pipeline` / `family_handler` classes at plugin
load time:

1. `_add_image_reference` / `_add_video_reference` are wrapped to recognize a
   tiny sentinel object carrying a precomputed latent; when they see one they
   append it straight into `refs` / `visual_latents` (skipping the pixel
   resize + VAE encode a live reference goes through), and otherwise fall
   through to the original, unmodified implementation.
2. `generate()` is wrapped so that, just before calling the original, it
   reads a small JSON blob out of Wan2GP's own generic `custom_settings`
   channel (already plumbed end-to-end from a submitted task to
   `pipeline.generate(**kwargs)` for several other models) and turns it into
   sentinel objects appended to `input_ref_images` / `input_frames` /
   `input_frames2` -- the same public parameters a live reference uses. This
   is what lets a RefMod apply even when you supply *no* live reference at
   all, which is the entire point of the feature.
3. The same wrapper also recognizes a second `custom_settings` key that means
   "this call is a RefMod extraction, not a real render": it runs the VAE
   encode/compression directly and returns `None` immediately, which is
   exactly what `generate()` already does when a user aborts mid-generation
   -- so nothing downstream needs to change to handle it gracefully.
4. `family_handler.query_model_def` is wrapped to add two "text"
   `custom_settings` entries (ids `h3_refmod_state` / `h3_refmod_extract`,
   the two keys used above) to MiniMax H3's model definition. **This one
   isn't optional.** Wan2GP validates every submitted task's
   `custom_settings` against the ids the target model declares (via
   `collect_custom_settings_from_inputs`, called from `validate_settings`
   for *every* task, including ones submitted through the API/plugin path)
   and silently replaces anything else with `None` -- so without this,
   points 2 and 3 above would build a correct payload that then gets wiped
   out one step later, with no error anywhere to explain why generation
   just... ran normally instead of doing what was asked. (This was exactly
   the bug in the first published version of this plugin: extraction quietly
   ran a full render instead of saving a mod, with nothing in the logs
   pointing at the real cause.)
5. Wan2GP builds its entire model catalog (`models_def`, what
   `get_model_def()` reads from -- a plain dict) **once, at import time,
   before any plugin is loaded.** Patching `query_model_def` (point 4) has
   no effect on entries computed before the patch existed -- it only affects
   *future* calls. So `plugin.py`'s `post_ui_setup` (which runs once
   Wan2GP has finished injecting plugin globals, still before the app starts
   serving requests) calls Wan2GP's own `refresh_model_defs()` -- its
   supported way to rebuild that catalog on demand -- once, forcing every
   MiniMax H3 model definition to be recomputed through the now-patched
   function. (This was the second bug found while fixing the first one: the
   `query_model_def` patch alone was necessary but not sufficient, since it
   never got a chance to run before the catalog it was supposed to affect
   had already been built.)
6. `prepare_inputs_dict` is wrapped (also via `post_ui_setup`, using
   `self.set_global`) to make the *inline* Media Generator panel work --
   see "How the inline panel works" above for the full explanation. Its
   patch is scoped as narrowly as possible (only touches its result for
   MiniMax H3 Ref2VA model types with a stashed selection) and wrapped in
   its own `try/except` (any failure leaves the original result untouched).
7. `_as_video` (a plain module-level helper in `pipeline.py`) is wrapped so
   it passes a video-kind RefMod sentinel through unchanged instead of
   crashing on it. `generate()` runs every entry of `input_frames`/
   `input_frames2` through this function itself, *before* looping over them
   to call `_add_video_reference` -- so the sentinel has to survive this
   call too, not just the one inside `_add_video_reference`, or it fails
   with `'_RefModVideoSentinel' object has no attribute 'ndim'` before point
   1's patch ever gets a chance to run.
8. Right after that, `generate()` also computes a total-duration budget
   check (`sum(video.shape[1] for video in video_sources) / fps`, enforcing
   Ref2VA's 15-second reference cap) before ever reaching
   `_add_video_reference`. The video-kind sentinel exposes a `.shape`
   property for exactly this -- a plausible reconstructed pixel-space shape
   derived from the latent's own dimensions (undoing the video VAE's causal
   4:1 temporal compression and 16x spatial downsampling), not real pixel
   data, since a RefMod has none to offer.
9. `validate_generative_settings` (a third staticmethod on the same
   `family_handler` class as point 4, patched alongside it) is wrapped so an
   audio reference doesn't get rejected as "0 visual references" just
   because the visual side is coming entirely from RefMods. This check runs
   *before* `pipeline.generate()` is ever called, straight off the raw
   native form fields (`image_refs`/`video_guide`), with zero visibility
   into RefMods -- so without this, combining an audio reference with
   RefMod-only visuals (no live reference image/video) would always be
   rejected by this pre-flight check, even though the RefMods would have
   supplied enough visual references once actually injected. The patch
   calls the original function first, unchanged, and only steps in for this
   one specific failure message -- recounting visual references with
   RefMods included and clearing the error if that's now enough; every
   other rule the original function enforces (durations, per-type caps,
   control-video-specific checks) is left completely untouched.
10. `_resize_video` (another module-level helper in `pipeline.py`, present
   in newer Wan2GP builds) is wrapped to pass a video-kind sentinel through
   untouched. Newer versions resize each reference video to the output
   resolution right before `_add_video_reference`; a RefMod carries an
   already-VAE-encoded latent rather than pixels, so there is nothing to
   bicubic-resize (and no pixel tensor to `.permute()`) -- the latent goes
   into the packed sequence at its own saved resolution. On older builds
   without this helper the patch is simply skipped.
11. `get_model_settings` (a plain wgp.py function, patched via
   `self.set_global` like `prepare_inputs_dict`) closes a gap that one
   doesn't cover: clicking **Generate** on the Media Generator page doesn't
   call `prepare_inputs_dict` again -- it reads the task straight out of
   `get_model_settings(state, model_type)`, a cache last refreshed whenever
   `save_inputs`/`prepare_inputs_dict` most recently ran, which only
   happens when a *native* form field changes. If the last thing touched
   before clicking Generate was a RefMod picker in the inline panel and
   nothing else, that cache could be one selection behind. This wraps
   `get_model_settings` to re-apply the same freshest-`state[STASH_KEY]`
   injection one more time, right at the point the task is actually
   assembled -- the last possible moment before it's queued.
12. `_load_audio_reference` and `_add_audio_reference` (both bound methods
   on `MiniMaxH3Pipeline`, patched the same way as their image/video
   counterparts) let an audio-kind RefMod's sentinel pass through and skip
   re-encoding, exactly mirroring points 2 and 3 above -- `generate()` calls
   `_load_audio_reference(audio_guide)` inline, *before* `_add_audio_reference`
   ever sees it, so the loader needs the same "pass a sentinel through
   untouched" treatment `_as_video` gets for video.
12b. `_prepare_audio_references` (present in newer Wan2GP builds; patched
   only if found) closes a gap the point above doesn't: some builds route
   every audio reference through this function first, which computes a
   *shared* 15-second duration budget across all audio sources up front
   (checking `torch.is_tensor(source)` vs. `sf.info(source).duration`,
   which crashes on a plain sentinel object -- neither a real waveform nor
   a file path) and can truncate whichever "waveform" it produces per
   source if the combined total goes over. A `torch.Tensor` subclass isn't
   a safe fix here either: the function's own truncation slicing on a real
   waveform isn't guaranteed to preserve a custom subclass or a `.latent`
   attribute across the op, which would silently detach the real data from
   the result. The patch instead mirrors the original function's exact
   duration/truncation logic, with a dedicated `isinstance`-checked branch
   for RefMod sentinels that computes duration from (and, if needed,
   truncates) the real `.latent` tensor's own time axis directly, then
   re-wraps the result back into a sentinel so `_add_audio_reference`
   downstream still recognizes it. Every non-sentinel source is delegated
   to the untouched original per-source logic, unchanged.
13. `_inject_refmods` itself checks `kwargs.get("window_no")` before doing
   anything else, and skips RefMod injection entirely when it's an int
   greater than 1. Wan2GP's own sliding-window loop (also used by
   **Continue Video**) calls `generate()` once per window, passing
   `custom_settings` (carrying the RefMod selection) unconditionally every
   time -- but its own *native* references (`image_refs`, `prefix_video`,
   a reference video's own frames) are only fed into `window_no==1`; every
   later window continues from the previous window's own tail frames
   instead, never the original reference again. Without this check, a
   RefMod -- having no visibility into which window it's on from inside
   `generate()` itself -- would get re-injected as a "reference" on every
   single window, so a few frames of it would visibly appear at every
   window boundary in the output. A plain, single-window generation never
   sets `window_no` above 1 either, so this never affects normal use.

14. `_prepare_condition_rows` is wrapped to lift Wan2GP's own reference
   caps for RefMods. `generate()` enforces "at most 12 references: 9
   images, 2 videos, 2 audio clips" *inline*, between building the
   reference list and using it -- a UI/product limit, not an architectural
   one (MiniMax H3 uses runtime-computed RoPE positions, an unbounded
   reference loop, and free-running `<Picture N>` labels; the ComfyUI
   community has verified 15 image references working). RefMod latents are
   always appended immediately, in natural order. Only the `refs` entries
   of RefMods that would push the check over its limits are held back from
   it, then reinserted **at their original positions** from
   `_prepare_condition_rows`, which runs right after the check. Keeping
   natural order matters: the latent-upscaler refinement pass (phase 2)
   re-adds references into fresh lists and aligns them with phase 1's by
   position. As many RefMods as fit stay visible to the check, so the
   separate "at least as many visual as audio references" rule still holds
   when a live audio clip relies on RefMod visuals. Video/audio RefMods
   beyond Wan2GP's two native kwargs each are added directly, in both
   phases. Live (non-RefMod) references are still counted normally.

15. Newer Wan2GP builds expose a **third** native reference-video slot
   (`input_frames3`, flag `*`) and a third audio one (`audio_guide3`, flag
   `D`), and raised their own caps to 9 images / 3 videos / 3 audio. The
   plugin detects which of these kwargs `generate()` actually accepts and
   fills them before falling back to direct injection, so it works on
   builds with either two or three native slots.
16. Those builds also share a 15-second budget between reference videos by
   trimming them (`video[:, :max_frames]`, when `-` is in
   `video_prompt_type` -- a flag older settings get migrated to
   automatically). A RefMod carries a latent, not pixels, so the video
   sentinel implements that one slicing form: the pixel-frame count is
   converted back to latent frames (undoing the causal 4:1 temporal
   compression) and a trimmed sentinel is returned. Without it, two or more
   long video RefMods raise "'_RefModVideoSentinel' object is not
   subscriptable".

None of this edits any file inside your Wan2GP install; it's applied purely
in-memory, once, and is safe to apply twice (idempotent) if the plugin is
reloaded.

## Known limitations / not (yet) ported

Compared to the ComfyUI pack, this version does **not** include:

- The A/B "axis" loader (two mods on one signed slider).
- Bulk folder extraction.
- Saved/shareable curve-graph PNG presets.
- The per-denoising-step curve wrapper (`H3 RefMod Step Curve`) -- only the
  per-frame curve (baked in once, at injection time) is available here.
- More than 2 simultaneous **video-kind** mods in one generation, at most 2
  **audio-kind** mods, and at most 9 **image-kind** mods (12 total) -- these
  are Wan2GP's own native Ref2VA reference limits, not an extra restriction
  added by this plugin. Video and audio each have their **own**, separate
  15-second budget -- they don't share one.
  **A mod extracted from several stacked images counts once per image it
  contains**, not once per mod: a mod trained on 5 images uses 5 of the 9
  available image slots by itself. This isn't a soft cap this plugin could
  safely raise -- MiniMax H3's own packing code assigns a single-frame
  position grid to every "image" reference, so feeding it more than one
  frame either crashes or leaves every frame at the exact same position
  (silently indistinguishable to the model), rather than degrading
  gracefully. A live counter above the mod pickers (**Images: X / 9**,
  **Video: ~Xs / 15s**, **Audio: Xs / 15s**) tracks this in real time,
  accounting for strength, copies, and each selected mod's own frame count,
  so you don't have to do the math by hand or discover an overflow only
  when generation fails.
- The inline panel (and this plugin's own Generate tab) offer 9 image-kind +
  2 video-kind + 2 audio-kind mod slots -- matching MiniMax H3 Ref2VA's own
  native caps on each, so there's no situation where a slot is available but
  couldn't possibly be used.
- The inline panel hides automatically after a model switch (see "How the
  inline panel works" below), but stays visible right after Wan2GP starts
  if a MiniMax H3 Ref2VA model was already selected and nothing else has
  been switched yet -- there's no confirmed signal for "the page just
  loaded with model X" to react to, only for an actual switch.

Each selected video-kind mod goes into its own native "Reference/Control
Video" slot -- the exact same mechanism MiniMax H3 Ref2VA's own "Use Two
Reference Videos" option uses, one mod per slot -- rather than being merged
into a single combined tensor. Since there are only 2 such slots, at most 2
video-kind mods can be used per generation (matching the 2 "Video Mod" rows
in the picker); if a live reference video from the main form already
occupies both slots, video-kind mods have nowhere left to go and generation
falls back to running without them (logged, not a hard failure). The two
slots are fully independent -- mods of different resolutions can be used
together without issue.

## Using a saved RefMod

Each of the up to 9 image + 2 video slots in the picker is just a mod name
and a **Strength** slider (0 to 2, default 1). That's the whole surface:
retention, curve, scramble-seed, and per-mod copies were part of earlier
versions of this plugin and have since been removed -- in testing they
added UI complexity without changing outcomes enough to be worth it for
most people, and a strength of 1 already means "use the mod as saved".
Strength above 1 doesn't clip at the mod's original encode -- it extrapolates
past it, which can push a subtle mod harder but can also push it into
artifacts; there's no ceiling built in beyond the slider's own 2.0 max, so
treat values past 1 as an experiment, not a default.

## Testing notes (please read before relying on this)

This plugin was written and unit-tested against Wan2GP's source code
directly, including a fully mocked stand-in for `MiniMaxH3Pipeline` that
exercises the exact extraction and injection code paths above end-to-end.
**It has not been run against the real ~33B/20B MiniMax H3 weights on a GPU**,
since that isn't possible in the environment this was written in. Please
verify, on your own machine, before relying on it:

- That extraction actually produces a plausible reference in a follow-up
  generation (compare a `strength=1.0`, `mode=encode` mod against feeding the
  same image as a live reference -- they should look close to identical).
- That mixing several mods, and the `retention`/curve controls, behave the
  way the tooltips describe.
- The console/log output during extraction and generation, if anything looks
  off -- every step in `patches.py` prints a `[H3RefMod] ...` line.

If `models.minimax_h3.pipeline` has moved or changed shape in your installed
Wan2GP version, `install_patches()` fails safe: it prints a clear message and
the plugin's tab still opens (with extraction/generation disabled) instead of
crashing Wan2GP's startup.

You may see the `[H3RefMod]` setup lines (patched pipeline, refreshed model
catalog, hooked `prepare_inputs_dict`) printed twice at startup -- Wan2GP
appears to call plugins' `post_ui_setup` more than once during its own
startup sequence. This is harmless (every patch here checks a marker before
re-applying itself) and only costs a fraction of a second rebuilding the
model catalog an extra time; it doesn't indicate anything went wrong.

## License / attribution

This plugin's `core.py` is a close port of `core.py` from
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)
by Luisa (luisacaotica), MIT License, (c) 2026. The rest of this plugin
(`storage.py`, `patches.py`, `plugin.py`, `encframes.py`) is new code written
for Wan2GP.

The encoder frames stored inside mod files, the option to send a multi-picture
mod as one video reference, and the choice of how many pictures the prompt
encoder is shown (fork.8) are inspired by the RefMod implementation in
[ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder)
by Adudeguyman. The code here is written for Wan2GP rather than copied, and
the stored frames use the same file keys (`enc_0`, `enc_1`, ... with
`enc_times` / `enc_fps`), so they carry over between the two tools.

Original Wan2GP port by [g3n3rativ3](https://github.com/g3n3rativ3).

Forked from: https://github.com/g3n3rativ3/MiniMaxH3Mod-for-WanGP

Fork maintained by [jedsdead](https://github.com/jedsdead): https://github.com/jedsdead/Refmod-Fork
