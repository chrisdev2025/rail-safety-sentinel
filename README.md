# Virtellect.ai

The marketing site for **Virtellect** — AI built for personal injury &amp; PIP practices.

It's a self-contained `index.html` plus one share image (`og-card.png`): no build step,
no framework, no external dependencies. Every font is a system stack and every icon is
inline SVG, so it loads instantly and works offline. That also makes it trivial to host
anywhere.

**Upload both files to the web root together.** `og-card.png` is the card that shows
when the link is texted, posted, or shared on LinkedIn/Facebook/iMessage — the page
references it at `https://virtellect.ai/og-card.png`.

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
- **Cloudflare Pages / S3 / any web server** — upload `index.html` and `og-card.png`.

Point the `virtellect.ai` domain at whichever host you choose.

## Wiring up the contact form (2 minutes)

The form is ready for real lead capture — it just needs an endpoint. It submits over AJAX
with a sending → success/error flow, includes a honeypot for spam, and (until you add an
endpoint) falls back to opening the visitor's email client.

1. Create a free form endpoint with any of these — no code changes needed for any of them:
   - **Formspree** — sign up, create a form, copy the URL `https://formspree.io/f/xxxxxxx`
   - **Web3Forms** — get an access key by email, use `https://api.web3forms.com/submit`
     and add `<input type="hidden" name="access_key" value="YOUR-KEY">` inside the form
   - **Basin** — `https://usebasin.com/f/xxxxxxxx`
   - **Getform** — `https://getform.io/f/xxxxxxx`
2. In `index.html`, find `var FORM_ENDPOINT = "";` (near the bottom, in the `<script>`) and
   paste your URL between the quotes.
3. That's it. Submissions now land in your dashboard/inbox and visitors see an inline
   "Request received" confirmation.

## Before you go live — quick checklist

- [ ] **Add your form endpoint** (see above). Left blank, the form opens the visitor's
      email client to `hello@virtellect.ai` instead of capturing the lead on a server.
- [ ] **Set the real inbox.** Replace `hello@virtellect.ai` everywhere with your address
      (it appears in the form fallback `CONTACT_EMAIL`, the contact section, and the footer).
- [ ] **Confirm the claims.** The security/handling language describes intended practices —
      make sure it matches what Virtellect actually does before publishing.
- [ ] **Add real proof** when you have it — client outcomes, a named case study, logos.
      The stat band is written as honest capability statements, not invented metrics.
- [ ] **Share image is included** — make sure `og-card.png` sits next to `index.html` at
      the web root. Test it by pasting the live URL into LinkedIn's Post Inspector or
      a text message after launch.
- [ ] **Optional:** swap the system serif for a licensed display face (e.g. a self-hosted
      `@font-face`) if you want a fully bespoke wordmark.

## Note on repository contents

This repo also contains an unrelated `sentinel.py` from before the site work began. It's
left untouched. Remove it if it's no longer needed.
