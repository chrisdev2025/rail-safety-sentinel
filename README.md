# Virtellect.ai

The marketing site for **Virtellect** — AI built for personal injury &amp; PIP practices.

It's a single, self-contained `index.html`: no build step, no framework, no external
dependencies. Every font is a system stack and every icon is inline SVG, so it loads
instantly and works offline. That also makes it trivial to host anywhere.

## Design at a glance

- **Concept** — "Counsel-grade AI." The whole page is structured like a case file,
  because a personal-injury professional's world *is* the file.
- **Palette** — deep oxblood (gravitas), burnished brass (the one bold accent, reserved
  for actions), warm parchment ground, ink. Deliberately zero AI-blue.
- **Type** — a refined serif for headlines, a clean sans for body, and monospace for
  docket numbers, statute references, and labels (the way legal documents actually read).
- **Themes** — light and dark are both first-class. Respects the visitor's OS setting and
  offers a toggle (remembered in `localStorage`).
- **Motion** — restrained: a hero load sequence, scroll reveals, and a one-time "records →
  chronology" sweep. All of it honors `prefers-reduced-motion`.

## Preview it locally

```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

Or just open `index.html` in a browser.

## Deploy

Because it's a static file, any static host works:

- **Vercel** — `vercel --prod` from this folder.
- **Netlify** — drag the folder onto the dashboard, or `netlify deploy --prod`.
- **GitHub Pages** — enable Pages on this repo/branch; `index.html` is served at the root.
- **Cloudflare Pages / S3 / any web server** — upload `index.html`.

Point the `virtellect.ai` domain at whichever host you choose.

## Before you go live — quick checklist

- [ ] **Wire up the contact form.** It currently hands off to `mailto:hello@virtellect.ai`.
      For real lead capture, point the form at a backend (Formspree, Basin, a Netlify form,
      or your own endpoint). Search `briefing-form` in `index.html`.
- [ ] **Set the real inbox.** Replace `hello@virtellect.ai` everywhere with your address.
- [ ] **Confirm the claims.** The security/handling language describes intended practices —
      make sure it matches what Virtellect actually does before publishing.
- [ ] **Add real proof** when you have it — client outcomes, a named case study, logos.
      The stat band is written as honest capability statements, not invented metrics.
- [ ] **Swap the OG image** — add a share image and set `og:image` in the `<head>`.
- [ ] **Optional:** swap the system serif for a licensed display face (e.g. a self-hosted
      `@font-face`) if you want a fully bespoke wordmark.

## Note on repository contents

This repo also contains an unrelated `sentinel.py` from before the site work began. It's
left untouched. Remove it if it's no longer needed.
