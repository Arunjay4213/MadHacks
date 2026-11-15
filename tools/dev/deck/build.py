# Builds the redesigned slide files from this script plus the existing speaker notes.
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'root', 'project')
NOTES = json.load(open(os.path.join(HERE, 'notes.json')))
EXTRA = {
    'clock': ' On the slide the 16 fps line is short: if a chunk takes longer than 250 ms, the video plays faster than it is made and has to stall.',
    'm-diffusion': ' The three frames are an illustration, not model output: the formula applied in pixel space to one of our live frames with sigma = 1, 0.83 and 0. The model itself adds and removes noise on latents, not pixels. We tried 3 and 4 steps: 277 and 349 ms per chunk instead of 218, and the picture looked worse.',
    'fix-drift': ' Why fresh noise failed: each refresh drew new noise, so it was a new sample near the saved frame and the car jumped around. Why a fixed seed works: noise seeded from the video and the saved frame returns to the same picture every time.',
    'fix-cut': ' In short: 3 ghosted chunks became 1 clean cut.',
    'testing': ' The eight frames are from the live run at 03:42 UTC; the two FAIL frames failed on "Is the picture free of severe defects?" (the car on the water) and "Is there a road?" (no road).',
}

B = lambda i: '/_blob/' + i
IMG = {
    'cover': B('333ddb1f3622b484b24a6091db3b4ba8'), 'demo': B('423cbc3fbc521ae60667b0b040c53aeb'),
    'hard': B('8cf158f4de5ce388b3a692c4d35fa389'),
    'meadow': B('5cc37610fab08b0b3247e108d4463c49'), 'winter': B('c01a48a6aa3eb2f1cf61f8c9e6e9001d'),
    'nlion': B('8941124e6e46a3a5a67b5617f7dcfe8d'), 'dlion': B('a0cae1acc26abc520a9260ea64be94da'),
    'csummer': B('61bdd76e29dea2885cecabefbfbb5685'), 'csnow': B('bdf97010ec97fd0e419fe7699341beb4'),
    'cdesert': B('152a12c83d1d48375f15d48aaaa6a1ff'), 'cneon': B('11e15b25efcb33ec7222565e615bd688'),
    'dissolve': B('9efd3a37a7d77c5d3fa16177889c1e5b'),
    'c05': B('17c3a542daf1b4c8a8feca5c2f843dae'), 'c30': B('29918585d3fb229f6357f3a4efa2bd08'), 'c60': B('3e9b232cdab3a6c3d7312f088292dd0f'),
    'fc_clean_car': B('3d11d110a3d3ef412fc93d1dfbe7ead8'), 'fc_clean_meadow': B('76f8f4c29feb206ccb97aaa2f2f56b63'),
    'fc_ghost_car': B('5583cd4bc8be40d78fbb00110363aa38'), 'fc_ghost_meadow': B('d394e8814d1fc7cd74f0fef8331f246d'),
    'fd_after': B('b2360a04d5b5e0b5bfa687bc8bb70141'), 'fd_before': B('1ab59aa724b86b99f5213f876b933636'),
    'water': B('36951700c99b0c3f725f34f93b9bc3cd'),
    'n1000': B('4e9090bd63e216b388c8734ed17802c2'), 'n500': B('a94ef4dcc077605688cc26886428fb98'), 'nclean': B('ed55eaaf21e65756d43900dee6c1a57b'),
    'team': B('e51ce0a8c50a84df27bd8500550df5c4'),
    'u_broken': B('269bd28fb1f1f0e50a7b14289fb8fd10'), 'u_drift': B('0f3a73e004a54091d47d51db28418084'),
    'u_ghost': B('0111c2a036b976d472063426247aed65'), 'u_snow': B('e049196531b00c4f78401d1ec22064e9'),
    'acc': [B(x) for x in ['1abc276c5dd9a2e1891130782659efad', '609e44a3fa5d08dd8cb5005143140172', '37c0173be5cd5c2d33f65a170da6503d', '9d8962bd7310d14105d56250937e241f',
                           'b8a072fad966d8864a7b6e0b660ea2a2', '97c123380269cec304b78c27a63f030d', '794b908d0f4fde8ed523a6c3723ccf19', '38af7d084b40deae733c69343022c320']],
}

# ---- math helpers -------------------------------------------------------
def it(s):
    out = []
    for ch in s:
        if 'a' <= ch <= 'z':
            out.append('ℎ' if ch == 'h' else chr(0x1D44E + ord(ch) - 97))
        elif 'A' <= ch <= 'Z':
            out.append(chr(0x1D434 + ord(ch) - 65))
        else:
            out.append({'α': '\U0001D6FC', 'σ': '\U0001D70E', 'ε': '\U0001D700'}.get(ch, ch))
    return ''.join(out)

def eq(size, parts, cls='', style=''):
    toks = []
    for p in parts:
        sub = p.startswith('_')
        txt = (p[1:] if sub else p).replace(' ', '&#160;')
        toks.append(f'<p class="mx{" sb" if sub else ""}">{txt}</p>')
    return f'<div class="eq eq{size} {cls}" style="{style}">' + ''.join(toks) + '</div>'

ARROW = lambda w=110, col='#4B463E', sw=3: (f'<svg aria-label="to" width="{w}" height="40" viewBox="0 0 {w} 40" style="width:{w}px; height:40px; flex:none">'
                                          f'<path d="M2 20 H{w-6} M{w-22} 6 L{w-4} 20 L{w-22} 34" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/></svg>')

def foot(text, style=''):
    return f'<p class="foot" style="{style}">{text}</p>'

def pn(style=''):
    return f'<div class="pn" style="{style}"></div>'

S = {}

# 1 ---- cover ---------------------------------------------------------------
thumbs = [('meadow', 'start', 'Live frame: a green meadow with one oak tree under a blue sky'),
          ('winter', '“make it winter”', 'Live frame: the same field in winter, snow on the ground, bare tree'),
          ('nlion', '“add a lion”', 'Live frame: snowy night under a frosted tree, a lion and a red car'),
          ('dlion', '“make it a desert”', 'Live frame: a desert with red cliffs, the lion and the red car kept')]
th = ''
for i, (k, c, a) in enumerate(thumbs):
    x = 988 + i * 204
    th += (f'<img class="in d{5+i}" src="{IMG[k]}" alt="{a}" style="position:absolute; left:{x}px; top:852px; width:192px; height:110px; object-fit:cover">'
           f'<p class="q in d{5+i}" style="position:absolute; left:{x}px; top:972px; width:200px; font-size:27px; color:var(--on2)">{c}</p>')
S['cover'] = f'''<section id="cover" class="s dark" style="padding:0" data-transition="fade">
<img class="settle" src="{IMG['cover']}" alt="Live frame: a snowy night under a frosted tree, with a lion and a red car" style="position:absolute; left:0; top:0; width:1920px; height:820px; object-fit:cover">
<p class="k in d2" style="position:absolute; left:128px; top:88px; width:900px; color:var(--on)">MadHacks 2026</p>
<h1 class="d in d3" style="position:absolute; left:128px; top:150px; width:900px">Mirage</h1>
<p class="t in d3" style="position:absolute; left:128px; top:300px; width:900px; font-size:64px; font-style:italic; color:var(--on)">Live voice-steered video</p>
<p class="lead in d4" style="position:absolute; left:128px; top:852px; width:760px; font-size:32px">A video drawn from pure noise, 16 frames a second, that changes while it plays when you speak.</p>
<p class="k in d5" style="position:absolute; left:128px; top:950px; width:800px; color:var(--on)">Arunjay Agrawal · Ishaan · Ani · Prateek</p>
<p class="cap in d5" style="position:absolute; left:128px; top:992px; width:840px; font-size:24px">One live run on one NVIDIA L40S · 672 × 384 · 15 Nov 2026, 00:35 UTC</p>
{th}'''

# 2 ---- credit ----------------------------------------------------------------
comps = [('Causal Forcing++', 'Real-time video model, 2 steps per frame. thu-ml, Apache-2.0.'),
         ('Wan2.1-T2V-1.3B', 'Base video model and umT5-xxl text encoder. Wan-AI, Apache-2.0.'),
         ('TAEHV', 'Tiny decoder from latents to pixels. madebyollin, MIT.'),
         ('LongLive', 'Rebuilding the frame cache at a prompt switch. NVIDIA.')]
cc = ''.join(f'<div class="rule" style="padding-top:20px; display:flex; flex-direction:column; gap:10px"><p class="h" style="font-size:38px">{n}</p><p class="cap" style="font-size:24px">{d}</p></div>' for n, d in comps)
stats = [('480 × 832', 'pixels per frame'), ('16 fps', 'frames per second'), ('2 × H100', 'GPUs'), ('0.69 s', 'end of speech to changed pixels')]
st = ''.join(f'<div style="display:flex; flex-direction:column; gap:12px"><p class="n3">{n}</p><p class="cap" style="font-size:24px">{d}</p></div>' for n, d in stats)
S['credit'] = f'''<section id="credit" class="s">
<p class="k in">Credit first</p>
<h2 class="t in d1" style="margin-top:28px; width:1560px">We build on <i>Training-Free Interactive World Models</i>,<br>by HRD Labs</h2>
<p class="lead in d2" style="margin-top:36px; width:1400px">They already steer a running video model with voice, on the same Causal Forcing++ base we use. Their numbers:</p>
<div class="in d3" style="display:flex; gap:88px; margin-top:36px">{st}</div>
<p class="cap in d4" style="margin-top:32px; font-size:24px">hrdlabs.org/research/interactive-world-models · code: github.com/bryandong24/SPED · We reuse their idea of steering the prompt embedding while the model generates.</p>
<div class="in d5" style="display:grid; grid-template-columns:repeat(4, 1fr); gap:48px; margin-top:auto">{cc}</div>
{foot('Sources: README.md (Credit), engine/MODEL_NOTES.md (licenses, LongLive), prior-art check of 15 Nov (HRD Labs specs as reported by the authors)')}
{pn()}'''

# 3 ---- added -----------------------------------------------------------------
items = [('A command layer', 'Scene state, named operations, undo, stacked changes and strength words (“a little snowy”).'),
         ('Free-form sentences', 'Claude Sonnet 5.5 turns any sentence into exact operations and one full scene description.'),
         ('A stream that stays clean', 'Re-anchor cuts, a clean-anchor refresh, a decoder restart and an 8-frame dissolve.'),
         ('Latency work', 'Command to screen, median <span class="acc">717 → 449 ms</span> for pinned sentences.'),
         ('All of it on one GPU', 'Generation, decoding and the text encoder share one NVIDIA L40S, at 384 × 672 and 16 fps.')]
rows = ''.join(f'<div class="rule in d{i+2}" style="display:flex; gap:32px; padding:20px 0 22px"><p class="n3 was" style="width:48px; flex:none; font-size:44px">{i+1}</p>'
               f'<div style="flex:1; display:flex; flex-direction:column; gap:6px"><p class="h" style="font-size:40px">{n}</p><p class="cap">{d}</p></div></div>' for i, (n, d) in enumerate(items))
S['added'] = f'''<section id="added" class="s alt" style="flex-direction:row; gap:96px">
<div style="width:520px; flex:none; display:flex; flex-direction:column">
<p class="k in">What this team added</p>
<h2 class="t in d1" style="margin-top:28px; font-size:104px; line-height:0.98">An editor for live video</h2>
<p class="body in d2" style="margin-top:auto; font-size:28px">No new model and no training: engineering on top of published methods.</p>
</div>
<div style="flex:1; display:flex; flex-direction:column">{rows}</div>
{foot('Sources: PLAN.md, README.md, board post 03:29 UTC 15 Nov (latency)')}
{pn()}'''

# 4 ---- hard (section opener) ----------------------------------------------------
S['hard'] = f'''<section id="hard" class="s dark" style="padding:0" data-transition="fade">
<img class="settle" src="{IMG['hard']}" alt="Live frame: a frosted tree at night under the moon, a red car parked in the snow" style="position:absolute; left:0; top:0; width:1920px; height:540px; object-fit:cover">
<p class="k in d2" style="position:absolute; left:128px; top:596px; width:800px">Why this is hard</p>
<p class="stmt in d3" style="position:absolute; left:128px; top:640px; width:1460px">Every <span class="acc">250 ms</span>, 1.3 billion parameters turn pure noise into 4 new frames.</p>
<p class="body in d5" style="position:absolute; left:128px; top:836px; width:1240px">Forever, at 16 frames a second. And when a sentence arrives in the middle of the stream, the film has to change, without breaking.</p>
{foot('Sources: engine/MODEL_NOTES.md (one call = one latent frame = 4 video frames = 0.25 s), engine/README.md')}
{pn()}'''

# 5 ---- clock -------------------------------------------------------------------
PX = 1664 / 250
g_end = 191.5 * PX; d_end = (191.5 + 21.9) * PX
ticks = ''
tl = ''
for ms in range(0, 251, 50):
    x = ms * PX
    ticks += f'<line x1="{x:.1f}" y1="30" x2="{x:.1f}" y2="44" stroke="#4B463E" stroke-width="1.5"/>'
    lab = f'{ms} ms' if ms == 250 else str(ms)
    if ms == 0:
        tl += f'<p class="cap" style="position:absolute; left:0; top:258px; width:120px; font-size:24px">0</p>'
    elif ms == 250:
        tl += f'<p class="cap" style="position:absolute; left:{1664-160}px; top:258px; width:160px; font-size:24px; text-align:right">250 ms</p>'
    else:
        tl += f'<p class="cap" style="position:absolute; left:{x-60:.0f}px; top:258px; width:120px; font-size:24px; text-align:center">{lab}</p>'
bar = (f'<svg aria-label="Budget bar: 191.5 ms generation, 21.9 ms decode, 36.6 ms spare, out of 250 ms" width="1664" height="48" viewBox="0 0 1664 48" style="position:absolute; left:0; top:206px; width:1664px; height:48px">'
       f'<line x1="0" y1="8" x2="1664" y2="8" stroke="#CFC6B6" stroke-width="1"/>'
       f'<rect x="0" y="0" width="{g_end-2:.1f}" height="16" fill="#1A1814"/>'
       f'<rect x="{g_end:.1f}" y="0" width="{d_end-g_end-2:.1f}" height="16" fill="#8E8576"/>'
       f'<rect x="{d_end+0.75:.1f}" y="0.75" width="{1664-d_end-1.5:.1f}" height="14.5" fill="none" stroke="#B5391B" stroke-width="1.5"/>'
       f'{ticks}</svg>')
S['clock'] = f'''<section id="clock" class="s">
<p class="k in">Hard, part 1: the clock</p>
<h2 class="t in d1" style="margin-top:28px">250 ms per chunk. Every chunk. Forever.</h2>
<div class="fade d2" style="position:relative; width:1664px; height:300px; margin-top:64px">
<p class="n3" style="position:absolute; left:0; top:0; width:600px">191.5 ms</p>
<p class="lab" style="position:absolute; left:0; top:92px; width:600px">Generate the next chunk</p>
<p class="n3" style="position:absolute; left:{g_end+14:.0f}px; top:0; width:140px; font-size:60px">21.9</p>
<p class="lab" style="position:absolute; left:{g_end+14:.0f}px; top:92px; width:140px">Decode</p>
<p class="n3 acc" style="position:absolute; left:{d_end+14:.0f}px; top:0; width:220px; font-size:60px">36.6</p>
<p class="lab" style="position:absolute; left:{d_end+14:.0f}px; top:92px; width:220px">Spare</p>
{bar}
{tl}
</div>
<p class="cap in d3" style="margin-top:8px">Decode turns the chunk into pixels. Spare is what is left before the viewer needs the frames.</p>
<div class="in d4" style="display:grid; grid-template-columns:repeat(3, 1fr); gap:64px; margin-top:auto">
<div class="rule" style="padding-top:22px; display:flex; flex-direction:column; gap:14px"><p class="n3">16 fps</p><p class="cap">4 frames per chunk, 0.25 s of video. A slower chunk makes the video stall.</p></div>
<div class="rule" style="padding-top:22px; display:flex; flex-direction:column; gap:14px"><p class="n3">3 passes</p><p class="cap">through the 1.3B network per chunk: 2 to remove noise, 1 to write the new frame into the model’s memory.</p></div>
<div class="rule" style="padding-top:22px; display:flex; flex-direction:column; gap:14px"><p class="n3">417 ms</p><p class="cap">per chunk at the model’s native 480 × 832 on this GPU. Too slow, so we run 384 × 672.</p></div>
</div>
{foot('Measured on one NVIDIA L40S. Sources: engine/README.md (live /stats, 00:15 UTC 15 Nov), engine/MODEL_NOTES.md (Gate 1 table)')}
{pn()}'''

# 6 ---- untrained ------------------------------------------------------------------
fails = [('u_drift', 'Drift', '60 s after a reset, with no refresh: the picture falls apart.', 'Meadow frame 60 s after a reset with no refresh: smeared sky and blocky grass'),
         ('u_ghost', 'Ghosting', 'A cut decoded as if it continued the old scene: two trees at once.', 'Winter tree with a ghosted double image of the old green tree'),
         ('u_snow', 'Ignoring the prompt', '“make it winter”: a white strip on a green summer meadow.', 'Green summer meadow with only a white strip, not winter'),
         ('u_broken', 'Collapse', 'Over-saturated colour, streaks across the sky, blocky leaves.', 'Over-saturated meadow with streaked sky and blocky tree')]
fc = ''.join(f'<div class="in d{i+3}" style="width:392px; display:flex; flex-direction:column; gap:14px"><img src="{IMG[k]}" alt="{a}" style="width:392px; height:290px; object-fit:cover">'
             f'<p class="h" style="margin-top:8px; font-size:40px">{n}</p><p class="cap">{c}</p></div>' for i, (k, n, c, a) in enumerate(fails))
S['untrained'] = f'''<section id="untrained" class="s dark">
<p class="k in">Hard, part 2: the model was never trained for this</p>
<h2 class="t in d1" style="margin-top:28px; font-size:68px">It learned to make one 5-second clip from one fixed prompt</h2>
<p class="body in d2" style="margin-top:20px">We run it for minutes and change its mind mid-stream. Real frames, before our fixes:</p>
<div style="display:flex; gap:32px; margin-top:48px">{fc}</div>
<p class="cap in d7" style="margin-top:auto; font-size:24px">Also seen live: a red car that drove backwards, in 1 of 3 seeds with a short prompt.</p>
{foot('Sources: engine/samples/reset_no_refresh_baseline.jpg, reanchor_decoder_restart_steer.jpg, gate2_live_make_it_winter.jpg, broken_meadow_frame.jpg; board post on the reversing car (machine 2, 3 seeds)')}
{pn()}'''

# 7 ---- compound -------------------------------------------------------------------
f_comp = eq('M', [it('p'), '(video | prompt) = ', '∏', '_' + it('i'), ' ', it('p'), '(chunk', '_' + it('i'), ' | chunk', '_1', ' … chunk', '_' + it('i') + '−1', ', prompt)'], 'in d2', 'margin-top:44px')
trip = [('c05', '5 s: clean', 'Clean meadow 5 s after a reset', ''), ('c30', '30 s: over-saturated', 'Meadow 30 s after a reset, sky turned cyan and grass blocky', ''),
        ('c60', '60 s: broken', 'Meadow 60 s after a reset, picture broken into smears', ' acc')]
tc = ''.join(f'<div class="in d{i+4}" style="display:flex; flex-direction:column; gap:14px"><img src="{IMG[k]}" alt="{a}" style="width:528px; height:286px; object-fit:cover"><p class="h{x}" style="font-size:36px">{n}</p></div>' for i, (k, n, a, x) in enumerate(trip))
S['compound'] = f'''<section id="compound" class="s">
<p class="k in">Hard, part 3: errors compound</p>
<h2 class="t in d1" style="margin-top:28px">Each chunk is built on the last ones</h2>
{f_comp}
<p class="body in d3" style="margin-top:24px; width:1560px">The model writes every new chunk while looking at the last 21 it made, kept in a cache. A tiny error is copied into the next chunk, and that copy is copied again.</p>
<div style="display:flex; gap:40px; margin-top:auto">{tc}</div>
{foot('One live run with no commands and no refresh. Sources: engine/samples/reset_no_refresh_baseline.jpg, engine/README.md (refresh flag), engine/MODEL_NOTES.md')}
{pn()}'''

# 8 ---- language ---------------------------------------------------------------------
lang = [('“make it a desert”', 'Gave a field of flowers, then a random girl.', 'The keyword path wrote “a meadow … looks a desert”: a prompt that contradicts itself.'),
        ('“add a lion”', 'Not understood. Nothing changed.', '“lion” was not in the keyword table. Once read, it was still missing or tiny until the prompt gave it a size and a place.'),
        ('a red car on a road', 'Drove backwards in 1 of 3 seeds.', 'A short prompt leaves the direction open. A long one with “driving away from the camera” went forward in 3 of 3.')]
lc = ''.join(f'<div class="rule2 in d{i+2}" style="padding-top:24px; display:flex; flex-direction:column; gap:14px"><p class="q">{q}</p><p class="body ink" style="font-size:30px">{o}</p><p class="cap" style="font-size:24px">{w}</p></div>' for i, (q, o, w) in enumerate(lang))
S['language'] = f'''<section id="language" class="s alt">
<p class="k in">Hard, part 4: language is open-ended</p>
<h2 class="t in d1" style="margin-top:28px; width:1500px">People say anything. The model only follows one exact scene description.</h2>
<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:64px; margin-top:56px">{lc}</div>
<p class="h in d6" style="margin-top:auto; font-size:42px; width:1560px">So every sentence has to become one long, exact description of the <i>whole</i> scene, that keeps what is already there, while the video keeps playing.</p>
{foot('Sources: board posts from 14 Nov (desert bug reproduced on main, “add a lion” dropped, reversing car: 3 seeds on machine 2); ARCHITECTURE-commands.md')}
{pn()}'''

# 9 ---- demo ----------------------------------------------------------------------------
S['demo'] = f'''<section id="demo" class="s dark" style="padding:0" data-transition="fade">
<img class="settle" src="{IMG['demo']}" alt="Live frame: the starting meadow, one oak tree under a blue sky" style="position:absolute; left:0; top:0; width:1920px; height:760px; object-fit:cover">
<p class="k in d2" style="position:absolute; left:128px; top:812px; width:600px">Live demo</p>
<p class="d in d3" style="position:absolute; left:128px; top:852px; width:760px; font-size:150px">Now, live.</p>
<p class="cap in d4" style="position:absolute; left:960px; top:808px; width:832px; font-size:26px">Nothing on the screen is a recording. Watch the delay badge: command to the first new frame.</p>
<p class="q in d5" style="position:absolute; left:960px; top:900px; width:832px; font-size:38px; line-height:1.25; color:var(--on)">make it winter · add a red car · make it night · add a lion · sunny · <span class="acc">undo</span> · make it a desert · your sentence</p>'''

# 10 ---- pipeline -------------------------------------------------------------------------
st8 = [(0, 0, '1 · Language', 'Voice or text', 'Chrome speech, push-to-talk. Sent after 200 ms of unchanged words.', 'L'),
       (432, 0, '2 · Language', 'Reader', '133 pinned sentences, else Claude Sonnet 5.5 (about 1.4 s).', 'L'),
       (864, 0, '3 · Language', 'Scene state', 'Named operations, undo history, one full prompt.', 'L'),
       (1296, 0, '4 · GPU', 'Text encoder', 'umT5-xxl: 512 × 4,096 numbers per prompt. Cached.', 'G'),
       (1296, 300, '5 · GPU', 'Steering', 'Re-anchor cut, clean-anchor refresh, 8-frame dissolve.', 'G'),
       (864, 300, '6 · GPU', 'The model', 'Causal Forcing++ (Wan2.1 1.3B), 2 steps, 190 to 210 ms a chunk.', 'G'),
       (432, 300, '7 · GPU', 'TAEHV decoder', 'About 21 ms per chunk of 4 frames.', 'G'),
       (0, 300, '8 · Page', 'Browser', 'JPEG frames on a websocket, through a Cloudflare tunnel.', 'P')]
lines = ''
for x, y, _, _, _, k in st8:
    if k == 'G':
        lines += f'<line x1="{x}" y1="{y+3}" x2="{x+368}" y2="{y+3}" stroke="#1A1814" stroke-width="6"/>'
    elif k == 'L':
        lines += f'<line x1="{x}" y1="{y+3}" x2="{x+368}" y2="{y+3}" stroke="#1A1814" stroke-width="1.5"/>'
    else:
        lines += f'<line x1="{x}" y1="{y+3}" x2="{x+368}" y2="{y+3}" stroke="#1A1814" stroke-width="1.5" stroke-dasharray="8 6"/>'
def harrow(x1, x2, y):
    d = 1 if x2 > x1 else -1
    return (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#4B463E" stroke-width="1.5"/>'
            f'<path d="M{x2-12*d} {y-7} L{x2} {y} L{x2-12*d} {y+7}" fill="none" stroke="#4B463E" stroke-width="1.5"/>')
for a in (0, 432, 864):
    lines += harrow(a + 380, a + 420, 3)
for a in (1296, 864, 432):
    lines += harrow(a - 12, a - 52 + 0, 303)
lines += f'<path d="M1376 3 H1652 Q1662 3 1662 13 V291 Q1662 301 1652 301" fill="none" stroke="#4B463E" stroke-width="1.5"/>'
lines += f'<path d="M1650 287 L1662 301 L1674 287" fill="none" stroke="#4B463E" stroke-width="1.5"/>'
svg = f'<svg aria-label="Loop from voice to pixels: steps 1 to 4 left to right, then steps 5 to 8 right to left" width="1680" height="320" viewBox="0 0 1680 320" style="position:absolute; left:0; top:0; width:1680px; height:320px">{lines}</svg>'
stdiv = ''.join(f'<div style="position:absolute; left:{x}px; top:{y+26}px; width:340px; display:flex; flex-direction:column; gap:8px"><p class="lab">{l}</p><p class="h" style="font-size:40px">{n}</p><p class="cap" style="font-size:24px">{d}</p></div>' for x, y, l, n, d, k in st8)
S['pipeline'] = f'''<section id="pipeline" class="s">
<p class="k in">How it works</p>
<h2 class="t in d1" style="margin-top:28px">From your voice to pixels</h2>
<div class="fade d2" style="position:relative; width:1664px; height:520px; margin-top:64px">
{svg}
{stdiv}
</div>
<p class="cap in d4" style="margin-top:auto; font-size:24px">Thin line: language, on the CPU and Azure AI Foundry. &#160;Heavy line: one GPU, an NVIDIA L40S in development. &#160;Dashed: the page.</p>
{foot('Sources: engine/README.md (live flags, measured numbers), engine/MODEL_NOTES.md (conditioning shape), ARCHITECTURE-commands.md, web/README.md')}
{pn()}'''

# 11 ---- commands ----------------------------------------------------------------------------
tiers = [('Pinned', '133 likely demo sentences, answered in 4 to 9 ms.'),
         ('Otherwise', 'one Claude Sonnet 5.5 call returns the operations and a full scene description. About 1.4 s.'),
         ('If it times out', 'a keyword table (39 concepts, 281 trigger words) and a MiniLM vector matcher.'),
         ('Undo and reset', 'never wait for the language model.')]
tr = ''.join(f'<div class="rule" style="padding-top:14px; display:flex; flex-direction:column; gap:6px"><p class="lab ink">{a}</p><p class="cap" style="font-size:24px">{b}</p></div>' for a, b in tiers)
S['commands'] = f'''<section id="commands" class="s">
<p class="k in">The command layer</p>
<h2 class="t in d1" style="margin-top:28px">An editor, not a prompt box</h2>
<div style="display:flex; gap:96px; margin-top:48px; flex:1">
<div style="width:1040px; flex:none; display:flex; flex-direction:column">
<div class="in d2">
<table class="tbl">
<tr><th style="width:36%">You say</th><th style="width:34%">Operation</th><th style="width:30%">On screen</th></tr>
<tr><td>“make it winter”</td><td>add: season, winter</td><td>chip: winter</td></tr>
<tr><td>“add a lion”</td><td>add: object, lion</td><td>chip: lion</td></tr>
<tr><td>“make it a little snowy”</td><td>add: winter, strength 0.3</td><td>frosted grass</td></tr>
<tr><td>“remove the lion”</td><td>remove: lion</td><td>only the lion goes</td></tr>
<tr><td>“undo”</td><td>back one step</td><td>the scene before</td></tr>
<tr><td>“make it a desert”</td><td>new world, objects kept</td><td>snow gone, car and lion stay</td></tr>
</table>
</div>
<div class="in d4" style="display:flex; align-items:center; gap:40px; margin-top:auto"><p class="n2 acc" style="flex:none">14 / 14</p><p class="cap">commands understood in the live stacking test, with the chips right after every step, including 3 undos and “remove the lion”.</p></div>
</div>
<div class="in d3" style="flex:1; display:flex; flex-direction:column; gap:22px">
<p class="h" style="font-size:40px">How a sentence is read</p>
{tr}
</div>
</div>
{foot('Sources: ARCHITECTURE-commands.md, engine/README.md (Sonnet timings), engine/demo_prompts.json (133 entries), commands/keywords.json, PLAN.md Status (Gate 3)')}
{pn()}'''

# 12 ---- m-diffusion -----------------------------------------------------------------------------
f_diff = eq('XL', [it('x'), '_' + it('t'), ' = ', it('α'), '_' + it('t'), ' · ', it('x'), '_0', ' + ', it('σ'), '_' + it('t'), ' · ', it('ε')])
notes_d = [(eq('S', [it('x'), '_0'], style='width:170px; flex:none'), 'a clean frame'),
           (eq('S', [it('ε')], style='width:170px; flex:none'), 'random noise'),
           (eq('S', [it('α'), '_' + it('t'), ' = 1 − ', it('σ'), '_' + it('t')], style='width:170px; flex:none'), 'for this model, so t = 1000 is pure noise')]
nd = ''.join(f'<div class="rule" style="display:flex; align-items:baseline; gap:20px; padding-top:10px">{e}<p class="cap" style="font-size:24px">{t}</p></div>' for e, t in notes_d)
nrow = [('n1000', 't = 1000', 'Pure noise. Nothing of the scene yet.', 'Pure random noise'),
        ('n500', 't = 500', 'The grid is shifted: still about 83% noise.', 'The meadow frame mixed with 83 percent noise, barely visible'),
        ('nclean', 'Clean frame', 'Pass 3 writes it into the cache for the next chunk.', 'The clean meadow frame')]
nr = ''
for i, (k, h, c, a) in enumerate(nrow):
    nr += f'<div class="in d{i+4}" style="width:440px; display:flex; flex-direction:column; gap:10px"><img src="{IMG[k]}" alt="{a}" style="width:440px; height:206px; object-fit:cover"><p class="h" style="font-size:34px">{h}</p><p class="cap" style="font-size:24px">{c}</p></div>'
    if i < 2:
        nr += f'<div class="in d{i+4}" style="width:172px; height:206px; display:flex; align-items:center; justify-content:center">{ARROW(96)}</div>'
S['m-diffusion'] = f'''<section id="m-diffusion" class="s">
<p class="k in">The math, 1 of 4: diffusion</p>
<h2 class="t in d1" style="margin-top:28px">Drawing a frame out of noise, in 2 steps</h2>
<p class="lead in d2" style="margin-top:28px">Training mixes noise into real video. The model learns to take it back out.</p>
<div class="in d3" style="display:flex; align-items:center; justify-content:space-between; margin-top:20px">
{f_diff}
<div style="width:680px; display:flex; flex-direction:column; gap:6px">{nd}</div>
</div>
<div style="display:flex; margin-top:auto">{nr}</div>
{foot('Frames: the formula applied in pixel space to one of our frames, as an illustration. Sources: engine/MODEL_NOTES.md (schedule, warped t = 833 for “500”), steering/README.md (Tried and rejected), board post 23:53 UTC 14 Nov')}
{pn()}'''

# 13 ---- m-scale ------------------------------------------------------------------------------
ladder = [('384 × 672', 'pixels in each frame we show'), ('48 × 84', 'latent grid: the autoencoder shrinks each side by 8, keeping 16 channels'),
          ('1,008', 'tokens per latent frame: 2 × 2 patches make a 24 × 42 grid'), ('4 / s', 'latent frames per second: one latent frame becomes 4 video frames'),
          ('21,168', 'tokens of memory each new frame can look at: 21 frames × 1,008')]
ld = ''.join(f'<div class="rule in d{i+2}" style="padding-top:20px; display:flex; flex-direction:column; gap:14px"><p class="n3" style="font-size:60px">{n}</p><p class="cap" style="font-size:24px">{c}</p></div>' for i, (n, c) in enumerate(ladder))
S['m-scale'] = f'''<section id="m-scale" class="s alt">
<p class="k in">The math, 2 of 4: scale</p>
<h2 class="t in d1" style="margin-top:28px">One second of video, in numbers</h2>
<div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:40px; margin-top:52px">{ld}</div>
<div class="in d7" style="display:flex; gap:64px; margin-top:auto; align-items:flex-end">
<div style="width:1080px; flex:none; display:flex; flex-direction:column; gap:12px">
<div style="display:flex; align-items:baseline; gap:24px"><p class="h" style="font-size:40px">Rough compute per second of video</p><p class="cap" style="font-size:24px"><i>Estimate, not a measurement.</i></p></div>
<div style="display:flex; align-items:baseline; gap:20px; margin-top:8px"><p class="lab" style="width:190px; flex:none">Weights</p>{eq('S', ['2 × 1.3·10⁹ × 1,008 ≈ 2.6 TFLOP per pass'], style='flex:1')}</div>
<div style="display:flex; align-items:baseline; gap:20px"><p class="lab" style="width:190px; flex:none">Attention</p>{eq('S', ['4 × 30 × 1,536 × 1,008 × 21,168 ≈ 3.9 TFLOP per pass'], style='flex:1')}</div>
<div style="display:flex; align-items:baseline; gap:20px"><p class="lab" style="width:190px; flex:none">Per second</p>{eq('S', ['× 3 passes × 4 chunks per second'], style='flex:1')}</div>
</div>
<div style="flex:1; display:flex; flex-direction:column; gap:14px">
<div style="display:flex; align-items:baseline; gap:14px"><p class="mx acc" style="font-size:96px; line-height:1">≈</p><p class="num acc">78</p><p class="unit">TFLOP</p></div>
<p class="cap" style="font-size:24px">about 78 trillion operations for each second of video, done in real time</p>
</div>
</div>
{foot('Sources: engine/MODEL_NOTES.md (size, 4 frames per latent, window 21), Wan2.1-T2V-1.3B config (30 layers, width 1,536); compute is our estimate')}
{pn()}'''

# 14 ---- m-steer ------------------------------------------------------------------------------
def ease(r): return r ** 3 * (10 - 15 * r + 6 * r * r)
pts = ' '.join(f'{40 + 540 * (i / 60):.1f} {330 - 300 * ease(i / 60):.1f}' for i in range(61))
curve = (f'<svg aria-label="Min-jerk ease curve from 0 to 1, flat at both ends, against a dashed straight line" width="600" height="360" viewBox="0 0 600 360" style="position:absolute; left:0; top:44px; width:600px; height:360px">'
         f'<line x1="40" y1="330" x2="584" y2="330" stroke="#8E8576" stroke-width="1.5"/><line x1="40" y1="330" x2="40" y2="20" stroke="#8E8576" stroke-width="1.5"/>'
         f'<line x1="40" y1="30" x2="584" y2="30" stroke="#CFC6B6" stroke-width="1"/>'
         f'<line x1="40" y1="330" x2="580" y2="30" stroke="#8E8576" stroke-width="1.5" stroke-dasharray="7 7"/>'
         f'<polyline points="{pts}" fill="none" stroke="#1A1814" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/></svg>')
S['m-steer'] = f'''<section id="m-steer" class="s">
<p class="k in">The math, 3 of 4: steering</p>
<h2 class="t in d1" style="margin-top:28px">Steering the prompt, token by token</h2>
<div style="display:flex; gap:80px; margin-top:40px">
<div style="width:984px; flex:none; display:flex; flex-direction:column; gap:14px">
<p class="body in d2" style="font-size:28px">A prompt becomes 512 vectors of 4,096 numbers each (umT5-xxl). A command moves every vector from the old prompt toward the new one.</p>
{eq('M', [it('e'), '(', it('r'), ') = slerp(', it('e'), '_old', ', ', it('e'), '_new', ', ease(', it('r'), '))'], 'in d3', 'margin-top:14px')}
{eq('M', ['ease(', it('r'), ') = ', it('r'), '³ (10 − 15', it('r'), ' + 6', it('r'), '²)'], 'in d3')}
<p class="cap in d3" style="font-size:24px">slerp turns the direction and blends the length of each vector. The min-jerk curve starts and stops gently.</p>
{eq('M', ['target = blend(', it('E'), '_without', ', ', it('E'), '_with', ', ', it('s'), '),   ', it('s'), ' ∈ (0, 1.3]'], 'in d4', 'margin-top:10px')}
<p class="cap in d4" style="font-size:24px">Strength words: “a little” = 0.3, “a lot more” = 1.3.</p>
</div>
<div class="fade d3" style="position:relative; width:600px; height:440px">
<p class="cap" style="position:absolute; left:40px; top:0; width:560px; font-size:24px">weight on the new prompt (0 to 1)</p>
{curve}
<p class="cap" style="position:absolute; left:40px; top:410px; width:544px; font-size:24px; text-align:right">progress through the move →</p>
</div>
</div>
<div class="rule in d5" style="margin-top:auto; padding-top:18px; display:flex; gap:32px"><p class="lab ink" style="width:240px; flex:none">What we learned</p><p class="cap" style="font-size:24px">Blending alone did not change the picture. The model kept copying its own past frames, so “make it winter” left a green meadow. Live, a command moves in 1 chunk and then re-anchors: the next slide.</p></div>
{foot('Sources: steering/blend.py, steering/ease.py, steering/README.md (Controller, Re-anchor policy), engine/MODEL_NOTES.md (conditioning [1, 512, 4096])')}
{pn()}'''

# 15 ---- m-cut (dark) --------------------------------------------------------------------------
steps = ['Clear the model’s memory of the old frames.', 'Draw the next frame from pure noise, under the new prompt, in 2 steps.',
         'Restart the decoder, so it does not mix in the old scene.', 'Dissolve from the last frame shown, over 8 frames (0.5 s).']
sp = ''.join(f'<div class="rule in d{i+2}" style="display:flex; gap:28px; padding:14px 0"><p class="n3 was" style="width:36px; flex:none; font-size:40px">{i+1}</p><p class="body" style="font-size:28px; color:var(--on)">{s}</p></div>' for i, s in enumerate(steps))
tlab = ''.join(f'<p class="cap" style="position:absolute; left:{160 + 320 * i - 80}px; top:880px; width:160px; text-align:center; font-size:24px">{t} ms</p>' for i, t in enumerate([0, 125, 250, 375, 500, 625]))
S['m-cut'] = f'''<section id="m-cut" class="s dark">
<img class="fade d5" src="{IMG['dissolve']}" alt="Six live frames 125 ms apart: a green meadow dissolves into a snowy field" style="position:absolute; left:0; top:690px; width:1920px; height:179px; object-fit:cover">
<p class="k in">The math, 4 of 4: the cut</p>
<h2 class="t in d1" style="margin-top:28px">Forget the past, then dissolve</h2>
<div style="display:flex; gap:96px; margin-top:36px">
<div style="width:820px; flex:none; display:flex; flex-direction:column">{sp}</div>
<div class="in d4" style="flex:1; display:flex; flex-direction:column; gap:10px">
{eq('M', ['frame', '_' + it('k'), ' = (1 − ', it('a'), ') · old + ', it('a'), ' · new'])}
{eq('M', [it('a'), ' = (', it('k'), ' + 1) / (', it('N'), ' + 1),  ', it('N'), ' = 8'])}
<div style="display:flex; align-items:center; gap:20px; margin-top:28px"><p class="n2 was" style="font-size:96px">67-79</p>{ARROW(80, '#A8A196', 2.5)}<p class="n2 acc" style="font-size:96px">10-11</p></div>
<p class="cap" style="font-size:24px">largest jump between two frames at a cut, without and with the dissolve (0 to 255; normal motion reaches 17)</p>
</div>
</div>
{tlab}
{foot('“make it winter” on the live stream, every second frame. Sources: steering/README.md (Cut fade, Tried and rejected), engine/MODEL_NOTES.md (Rebase), engine recording 00:35 UTC 15 Nov')}
{pn()}'''

# 16 ---- fix-drift (split, images right) -----------------------------------------------------------
S['fix-drift'] = f'''<section id="fix-drift" class="s" style="padding:128px 1000px 160px 128px">
<div style="position:absolute; left:1040px; top:0; width:880px; height:1080px; background:#0E0D0B"></div>
<img class="settle" src="{IMG['fd_before']}" alt="Before: meadow 60 s after reset with no refresh, broken into smears" style="position:absolute; left:1040px; top:0; width:880px; height:532px; object-fit:cover">
<img class="settle d2" src="{IMG['fd_after']}" alt="After: meadow 90 s after reset, clean and the same scene" style="position:absolute; left:1040px; top:548px; width:880px; height:532px; object-fit:cover">
<p class="lab" style="position:absolute; left:1040px; top:470px; width:440px; box-sizing:border-box; padding:16px 28px; background:#0E0D0B; color:#EDE8DE">Before · 60 s, no refresh</p>
<p class="lab" style="position:absolute; left:1040px; top:1018px; width:440px; box-sizing:border-box; padding:16px 28px; background:#0E0D0B; color:#EDE8DE">After · 90 s, still clean</p>
<p class="k in">Fix 1 of 3</p>
<h2 class="t in d1" style="margin-top:28px">Stop the drift<br>with a clean frame</h2>
<p class="body in d2" style="margin-top:28px">Every 24 chunks (6 s), restart from a clean frame saved 3 chunks after the last scene change, with the <i>same noise</i> every time.</p>
<div class="in d3" style="display:flex; align-items:center; gap:36px; margin-top:72px"><p class="num was" style="font-size:180px">30.4</p>{ARROW(96)}<p class="num acc" style="font-size:180px">10.9</p></div>
<p class="cap in d4" style="margin-top:24px">how much the scene re-rolls at each refresh (mean pixel change, 0 to 255). Normal motion over 1 s: 12.1.</p>
{foot('Measured on the L40S, 360 chunks, parked-car scene. Sources: engine/MODEL_NOTES.md (Refresh noise table), steering/README.md, engine/samples/drift_after_83_90s.jpg', 'width:832px; padding-right:110px')}
{pn('right:auto; left:840px')}'''

# 17 ---- fix-cut (split, images left) -----------------------------------------------------------------
grid = ''
for (col, row, k, a) in [(0, 0, 'fc_ghost_meadow', 'Before: the cut frame shows the new winter tree over a ghost of the old green tree'),
                         (1, 0, 'fc_clean_meadow', 'After: a clean winter tree on the cut frame'),
                         (0, 1, 'fc_ghost_car', 'Before: the red car blurred and washed out on the cut frame'),
                         (1, 1, 'fc_clean_car', 'After: the red car sharp on a snowy road on the cut frame')]:
    grid += f'<img class="fade d{2+col+row}" src="{IMG[k]}" alt="{a}" style="position:absolute; left:{24 + col * 436}px; top:{318 + row * 262}px; width:420px; height:240px; object-fit:cover">'
S['fix-cut'] = f'''<section id="fix-cut" class="s" style="padding:128px 128px 160px 1000px">
<div style="position:absolute; left:0; top:0; width:904px; height:1080px; background:#0E0D0B"></div>
<p class="lab" style="position:absolute; left:24px; top:268px; width:420px; color:#A8A196">Before</p>
<p class="lab" style="position:absolute; left:460px; top:268px; width:420px; color:#EDE8DE">After: decoder restart</p>
{grid}
<p class="cap" style="position:absolute; left:24px; top:842px; width:856px; font-size:24px; color:#A8A196">The cut frame, on the meadow and on the car.</p>
<p class="k in">Fix 2 of 3</p>
<h2 class="t in d1" style="margin-top:28px">A clean cut instead of a ghost</h2>
<p class="body in d2" style="margin-top:28px; font-size:28px">The cut frame is a first frame, but the decoder kept the old scene in its memory and decoded it as a continuation.</p>
<div class="in d3" style="display:flex; align-items:center; gap:36px; margin-top:72px"><p class="num was" style="font-size:168px">110</p>{ARROW(96)}<p class="num acc" style="font-size:168px">901</p></div>
<p class="cap in d4" style="margin-top:20px">sharpness of the cut chunk, meadow (its neighbours: about 2,050)</p>
<div class="rule in d5" style="display:flex; align-items:baseline; gap:24px; margin-top:28px; padding-top:18px"><p class="n3" style="font-size:60px; flex:none">71 → 361</p><p class="cap">the same on the car (neighbours: about 510)</p></div>
{foot('Sharpness = variance of the Laplacian. Measured on the L40S. Sources: steering/README.md (Decoder restart, PR #83), engine/samples/reanchor_decoder_restart_steer.jpg', 'left:1000px; width:792px; padding-right:110px')}
{pn()}'''

# 18 ---- latency ----------------------------------------------------------------------------------------
SC = 0.8
before = [('End of speech (voice only)', 0, 200, 'voice', '200'), ('Reader (pinned sentence)', 200, 9, 'o', '9'), ('Text encoder (cached)', 209, 1, 'o', '1'),
          ('Wait for chunk boundary', 210, 91, 'cut', '91'), ('Generate the cut chunk', 301, 345, 'cut', '345'), ('Decode + JPEG', 646, 19, 'o', '19'),
          ('Playout queue', 665, 50, 'cut', '50'), ('Network', 715, 40, 'o', '≈ 40')]
after = [('After: command to screen', 0, 449, 'after', '449 median'), ('After: the cut chunk', 0, 162, 'after', '162 (was 412)')]
rows_all = [(r, i * 48) for i, r in enumerate(before)] + [(r, 8 * 48 + 32 + i * 48) for i, r in enumerate(after)]
bars = ''; vals = ''; labs = ''
for (lab, s, w, kind, v), y in rows_all:
    x = s * SC; ww = max(w * SC, 3); cy = y + 17
    if kind == 'voice':
        bars += f'<rect x="{x+0.75:.1f}" y="{cy+0.75:.1f}" width="{ww-1.5:.1f}" height="12.5" fill="none" stroke="#8E8576" stroke-width="1.5" stroke-dasharray="5 4"/>'
    else:
        col = {'o': '#BDB3A2', 'cut': '#1A1814', 'after': '#B5391B'}[kind]
        bars += f'<rect x="{x:.1f}" y="{cy}" width="{ww:.1f}" height="14" fill="{col}"/>'
    vals += f'<p class="cap" style="position:absolute; left:{384 + x + ww + 12:.0f}px; top:{y + 6}px; width:240px; font-size:24px; font-variant-numeric:tabular-nums{"; color:var(--ink); font-weight:500" if kind != "o" and kind != "voice" else ""}">{v}</p>'
    labs += f'<p class="cap" style="position:absolute; left:0; top:{y + 6}px; width:360px; text-align:right; font-size:24px{"; color:var(--ink); font-weight:500" if kind in ("cut", "after") else ""}">{lab}</p>'
H = 8 * 48 + 32 + 2 * 48
grid_l = ''.join(f'<line x1="{ms*SC:.1f}" y1="0" x2="{ms*SC:.1f}" y2="{H}" stroke="#DDD5C7" stroke-width="1"/>' for ms in range(0, 801, 100))
wsvg = f'<svg aria-label="Latency waterfall: stage medians before, and the measured totals after" width="660" height="{H}" viewBox="0 0 660 {H}" style="position:absolute; left:384px; top:0; width:660px; height:{H}px">{grid_l}<line x1="0" y1="{8*48+16}" x2="660" y2="{8*48+16}" stroke="#CFC6B6" stroke-width="1"/>{bars}</svg>'
S['latency'] = f'''<section id="latency" class="s">
<p class="k in">Fix 3 of 3</p>
<h2 class="t in d1" style="margin-top:28px">Command to screen, before and after</h2>
<div style="display:flex; gap:72px; margin-top:52px">
<div style="width:1120px; flex:none; display:flex; flex-direction:column; gap:20px">
<div class="fade d2" style="position:relative; width:1120px; height:{H}px">{wsvg}{labs}{vals}</div>
<p class="cap in d4" style="font-size:24px">Before: stage medians in ms, 15 live commands at 00:55 UTC (server 555, page 596). Dark bars: the stages we cut. After: only the measured total and cut chunk; no new stage split exists.</p>
</div>
<div class="in d3" style="flex:1; display:flex; flex-direction:column; gap:14px">
<p class="lab">Median, command to screen</p>
<div style="display:flex; align-items:baseline; gap:14px; margin-top:6px"><p class="num acc">449</p><p class="unit">ms</p></div>
<p class="body ink" style="font-size:30px">was 717 ms. p90: 737 → 498 ms.</p>
<p class="cap rule" style="font-size:24px; padding-top:14px; margin-top:8px"><i>2-step cut</i> and a deferred cache pass: the cut chunk 412 → 162 ms, same GPU, same test.</p>
<p class="cap" style="font-size:24px"><i>Interrupt</i> the old chunk mid-flight. <i>Flush</i> old frames already queued.</p>
<p class="cap" style="font-size:24px">Pinned sentences. A new sentence adds about 2 s: one Claude call.</p>
</div>
</div>
{foot('Measured on the L40S. Sources: engine/README.md (latency stages, Cut latency), board posts 03:29 UTC (717 → 449) and Ishaan’s PR #187 check (412 → 162), web/README.md (200 ms)')}
{pn()}'''

# 19 ---- testing ----------------------------------------------------------------------------------------------
acc = [('make it a desert', 'PASS', '320 ms'), ('make it night', 'PASS', '381 ms'), ('make it autumn', 'PASS', '461 ms'), ('make it sunset', 'PASS', '380 ms'),
       ('make it rain', 'PASS', '378 ms'), ('take me to the beach', 'FAIL', '377 ms'), ('add a red car racing on a road', 'FAIL', '2438 ms'), ('undo the winter', 'PASS', '1622 ms')]
alts = ['red car on a desert highway', 'red car in a neon city at night', 'red car on an autumn road', 'red car on a road at sunset',
        'red car on a wet road in the rain', 'red car driving on ocean waves', 'meadow with a tree and a small red car, no road', 'two lions under a green tree']
ag = ''
for i, (c, r, d) in enumerate(acc):
    fail = r == 'FAIL'
    ag += (f'<div style="display:flex; flex-direction:column; gap:8px"><img src="{IMG["acc"][i]}" alt="Live frame: {alts[i]}" style="width:245px; height:138px; object-fit:cover">'
           f'<p class="cap" style="font-size:24px; line-height:1.25; color:var(--ink)">“{c}”</p>'
           f'<p class="lab" style="{"color:var(--ink); font-weight:600" if fail else ""}">{r} · {d}</p></div>')
S['testing'] = f'''<section id="testing" class="s alt">
<p class="k in">How we know it works</p>
<h2 class="t in d1" style="margin-top:28px">A judge that looks at the frames</h2>
<div style="display:flex; gap:72px; margin-top:40px; flex:1">
<div style="width:560px; flex:none; display:flex; flex-direction:column; gap:20px">
<div class="in d2" style="display:flex; flex-direction:column; gap:10px"><p class="n2 acc" style="font-size:88px">47 / 47</p><p class="cap" style="font-size:24px">answers right on 16 frames with known contents, 4 of them known-bad. In-sample: 45 of 47 before one question was reworded.</p></div>
<div class="rule in d3" style="padding-top:22px; display:flex; flex-direction:column; gap:10px"><p class="n2" style="font-size:88px">20 / 23</p><p class="cap" style="font-size:24px">steps passed live, and 19 of 23 in a second run. We checked every failure by eye: the judge was right.</p></div>
<div class="rule in d4" style="padding-top:22px; display:flex; flex-direction:column; gap:10px"><p class="n2" style="font-size:88px">969</p><p class="cap" style="font-size:24px">automated tests pass, including the sentence-to-prompt table.</p></div>
</div>
<div class="fade d3" style="flex:1; display:flex; flex-direction:column; gap:24px">
<div style="display:grid; grid-template-columns:repeat(4, 245px); column-gap:20px; row-gap:28px">{ag}</div>
<p class="cap" style="font-size:24px; margin-top:auto">Live run, 03:42 UTC. Claude Sonnet 5.5 answers yes/no questions about the frame 3 s after each command: “Is there snow on the ground?”</p>
</div>
</div>
{foot('Sources: README.md (Acceptance), PR #231 (calibration and live runs), engine/samples/acceptance_live_0342.jpg, board post 04:54 UTC 15 Nov (969 tests)')}
{pn()}'''

# 20 ---- rejected -------------------------------------------------------------------------------------------------
S['rejected'] = f'''<section id="rejected" class="s">
<p class="k in">What we threw away</p>
<h2 class="t in d1" style="margin-top:28px">Measured, then rejected</h2>
<div class="in d2" style="margin-top:48px">
<table class="tbl" style="font-size:26px">
<tr><th style="width:26%">We tried</th><th style="width:34%">What we measured</th><th style="width:40%">Why it is off</th></tr>
<tr><td>torch.compile</td><td>23% faster: 193 → 148 ms per chunk</td><td>Frames differ from the reference (max pixel difference 255). CUDA graphs crash.</td></tr>
<tr><td>Full Wan decoder</td><td>157 ms per chunk, against 21.5 ms</td><td>Almost the same picture, 7 times slower.</td></tr>
<tr><td>3 or 4 denoising steps</td><td>277 and 349 ms, against 218 ms</td><td>Over budget, and the picture looked worse.</td></tr>
<tr><td>Native 480 × 832</td><td>417 ms per chunk</td><td>Over the 250 ms budget on this GPU.</td></tr>
<tr><td>Re-cache memory only</td><td>The world never changed; 2.3 s stall</td><td>A full cut is needed to change the scene.</td></tr>
<tr><td>Keep-layout start (t = 800)</td><td>Tree moves 7 to 16 px, not up to 194 px</td><td>Winter only partly arrives on the meadow. Off live.</td></tr>
</table>
</div>
<div class="in d3" style="display:flex; align-items:center; gap:40px; margin-top:auto"><p class="n2 acc" style="flex:none">12</p><p class="body" style="font-size:28px">ways to change a scene were compared side by side. The one that runs live: a clean cut with a decoder restart and an 8-frame dissolve.</p></div>
{foot('All on the L40S. Sources: KNOWN_ISSUES.md (torch.compile, layout), steering/README.md (Tried and rejected), engine/MODEL_NOTES.md (Gate 1 table)')}
{pn()}'''

# 21 ---- overnight ----------------------------------------------------------------------------------------------------
GREP = ('84', 'also reviewed by the Greptile code-review bot')
ov = [('185', 'pull requests merged', ' acc'), ('617', 'commits on main', ''), ('846', 'automated tests', ''), (GREP[0], GREP[1], '')]
oc = ''.join(f'<div class="rule2 in d{i+2}" style="padding-top:20px; display:flex; flex-direction:column; gap:14px"><p class="num{a}" style="font-size:136px">{n}</p><p class="cap">{l}</p></div>' for i, (n, l, a) in enumerate(ov))
def gpu(lab, head, cap, d):
    return (f'<div class="in {d}" style="flex:1; background:var(--paper); padding:32px 40px 36px; display:flex; flex-direction:column; gap:12px">'
            f'<p class="lab">{lab}</p><p class="h">{head}</p><p class="cap">{cap}</p></div>')
S['overnight'] = f'''<section id="overnight" class="s alt">
<p class="k in">Four people directing AI agents</p>
<h2 class="t in d1" style="margin-top:28px">Built fast, checked at every step</h2>
<div style="display:grid; grid-template-columns:repeat(4, 1fr); column-gap:64px; margin-top:52px">{oc}</div>
<p class="k in d6" style="margin-top:auto">Two GPUs, side by side</p>
<div style="display:flex; align-items:center; margin-top:24px">
{gpu('Test bench · NVIDIA L40S', 'Agents measure it', 'Frame sheets, fixed seeds, a vision judge', 'd6')}
<div class="in d7" style="width:380px; flex:none; display:flex; flex-direction:column; align-items:center; gap:6px; padding:0 24px">
<p class="cap" style="font-size:24px; text-align:center; font-style:italic; white-space:nowrap">only measured changes go live</p>
{ARROW(220)}
</div>
{gpu('Live engine · NVIDIA L40S', 'People test it', 'By hand, all night', 'd8')}
</div>
{foot('Counted 17:55 UTC, 15 Nov: gh pr list --state merged, git log origin/main, def test_ in */tests/*.py, Greptile reviews per PR')}
{pn()}'''

# 22 ---- hardware ------------------------------------------------------------------------------------------------------
S['hardware'] = f'''<section id="hardware" class="s">
<p class="k in">Hardware</p>
<h2 class="t in d1" style="margin-top:28px">Built on two L40S. Shown on an RTX PRO 6000 Blackwell.</h2>
<div style="display:flex; gap:96px; margin-top:52px">
<div class="in d2" style="width:600px; flex:none; display:flex; flex-direction:column; gap:18px">
<p class="h" style="font-size:40px">Development</p>
<p class="body" style="font-size:28px">Two NVIDIA L40S: one runs the live engine, one runs tests and measurements.</p>
<p class="body ink" style="font-size:28px"><b>Every other number in this deck was measured on an L40S.</b> The one Blackwell number, below, was measured tonight on the demo machine.</p>
</div>
<div class="in d3" style="flex:1">
<table class="tbl" style="font-size:28px">
<tr><th style="width:36%">Spec sheet</th><th style="width:28%">L40S</th><th style="width:36%">RTX PRO 6000 Blackwell Server Edition</th></tr>
<tr><td>Memory</td><td>48 GB GDDR6</td><td>96 GB GDDR7</td></tr>
<tr><td>Memory bandwidth</td><td>864 GB/s</td><td>about 1.8 TB/s</td></tr>
</table>
</div>
</div>
<div class="rule2 in d4" style="display:flex; align-items:center; gap:48px; margin-top:auto; padding-top:28px"><p class="n2 acc" style="flex:none">~150 ms</p><p class="body" style="font-size:28px">per chunk at the model’s native 832 × 480 on the RTX PRO 6000 Blackwell, measured tonight, against a 250&#160;ms budget. The L40S measured 417&#160;ms at that size, so the full size needs the Blackwell.</p></div>
{foot('Sources: NVIDIA datasheets (L40S, RTX PRO 6000 Blackwell Server Edition), engine/MODEL_NOTES.md, Blackwell run of 15 Nov')}
{pn()}'''

# 23 ---- limits ----------------------------------------------------------------------------------------------------------
lim = [('A new sentence takes about 2 s,', 'for one Claude call. Pinned sentences take about 0.45 s.'),
       ('Every command recomposes the shot:', 'the camera and layout can jump. Our layout-keeping option is off live.'),
       ('Past about 5 objects', 'the model drops some, even when the chips are right.'),
       ('The scene changes only after you finish speaking.', 'Early guesses from partial words show nothing yet.'),
       ('Added objects can be small or at the edge.', 'No real faces: a named person becomes a description of their look.'),
       ('384 × 672,', 'not the model’s native 480 × 832, on the L40S.')]
li = ''.join(f'<p class="body in d{i+2}" style="font-size:28px"><b class="ink">{a}</b> {b}</p>' for i, (a, b) in enumerate(lim))
S['limits'] = f'''<section id="limits" class="s" style="padding:128px 968px 160px 128px">
<img class="settle" src="{IMG['water']}" alt="A red sports car driving on top of turquoise ocean waves" style="position:absolute; left:1080px; top:0; width:840px; height:1080px; object-fit:cover">
<p class="cap" style="position:absolute; left:1080px; top:888px; width:840px; height:192px; box-sizing:border-box; padding:28px 48px; background:#0E0D0B; color:#EDE8DE; font-size:24px">“take me to the beach” on the car scene: the car drives on the water. Our vision judge marks it FAIL, and it stays out of the demo.</p>
<p class="k in">Limits, stated plainly</p>
<h2 class="t in d1" style="margin-top:28px">What does not work yet</h2>
<div style="display:flex; flex-direction:column; gap:22px; margin-top:44px">{li}</div>
{foot('Sources: KNOWN_ISSUES.md, PLAN.md Status (Gates 3 and 4), README.md (Current state), PR #231 (live run 03:42 UTC)', 'width:824px; padding-right:110px')}
{pn('right:auto; left:832px')}'''

# 24 ---- next ---------------------------------------------------------------------------------------------------------------
nx = [('Keep the layout across commands', 'Already tested: the tree moves 7 to 16 px instead of up to 194 px. Winter must still fully arrive before it goes live.'),
      ('Change the scene while you speak', 'Guesses from partial words already reach the engine. Next: make a right guess visible before the sentence ends.'),
      ('Skip the language model when we can', 'When the keyword table fully understands a sentence, answer at once and save the 2 s Claude call.'),
      ('The full 480 × 832 picture', 'Needs generation at least 1.7 times faster than the L40S gave us: 417 ms measured, 250 ms allowed.')]
cars = [('csummer', 'red sports car on a country road, seen from behind'), ('csnow', 'the same car on a snowy forest road'),
        ('cdesert', 'the same car on a desert highway'), ('cneon', 'the same car in a neon city at night')]
nxc = ''.join(f'<div class="in d{i+2}" style="display:flex; flex-direction:column; gap:14px"><img src="{IMG[k]}" alt="Live frame: {a}" style="width:392px; height:224px; object-fit:cover">'
              f'<p class="h" style="font-size:36px; margin-top:12px">{t}</p><p class="cap" style="font-size:24px">{d}</p></div>' for i, ((k, a), (t, d)) in enumerate(zip(cars, nx)))
S['next'] = f'''<section id="next" class="s">
<p class="k in">What is next</p>
<h2 class="t in d1" style="margin-top:28px">Four steps from here</h2>
<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:32px; margin-top:52px">{nxc}</div>
{foot('Sources: KNOWN_ISSUES.md (layout numbers, early steering, Sonnet skip), engine/MODEL_NOTES.md (417 ms). Frames: live car chain, 03:43 UTC 15 Nov')}
{pn()}'''

# 25 ---- team -------------------------------------------------------------------------------------------------------------------
team = [('Arunjay Agrawal', 'Real-time engine, GPU machines, team lead'), ('Ishaan', 'Steering inside the diffusion model'),
        ('Ani', 'The web page and voice input'), ('Prateek', 'Command understanding and the pitch')]
tm = ''.join(f'<div class="rule in d{i+3}" style="padding-top:18px; display:flex; flex-direction:column; gap:8px"><p class="h" style="font-size:40px">{n}</p><p class="cap" style="font-size:24px">{r}</p></div>' for i, (n, r) in enumerate(team))
S['team'] = f'''<section id="team" class="s" style="padding:128px 1000px 128px 128px">
<img class="settle" src="{IMG['team']}" alt="Live frame: a red sports car on an autumn road between golden trees" style="position:absolute; left:1040px; top:0; width:880px; height:1080px; object-fit:cover">
<p class="k in">The team</p>
<p class="d in d1" style="margin-top:36px; font-size:150px">Thank you.</p>
<div style="display:grid; grid-template-columns:repeat(2, 1fr); column-gap:48px; row-gap:36px; margin-top:72px">{tm}</div>
<p class="body in d7" style="margin-top:auto; font-size:26px">Built on <i>Training-Free Interactive World Models</i> by HRD Labs, with Causal Forcing++, Wan2.1, umT5, TAEHV and LongLive. Thanks to MadHacks.</p>'''

for sid, html in S.items():
    note = NOTES[sid] + EXTRA.get(sid, '')
    assert len(note) <= 4000, sid
    out = html.rstrip() + f'\n<aside>{note}</aside>\n</section>\n'
    open(os.path.join(ROOT, 'slides', sid + '.html'), 'w').write(out)
print('wrote', len(S))
