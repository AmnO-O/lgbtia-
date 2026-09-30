import csv, os, re, glob

DATA_DIR = r"C:\CODE_SOMETHING\LGBT\LGBT"
BACKUP_DIR = os.path.join(DATA_DIR, "_backup_orig")
DESC_MAX = 800

STORED_KEYS = {
    "subscribe", "subscribe to", "subscription", "subscriber",
    "newsletter", "sign up", "signup", "register", "join us",
    "for more breaking news", "watch more", "click here", "read more:",
    "learn more", "get involved", "become a member", "like this video",
    "comment below", "share this", "follow us", "check out our website",
    "more at", "stay tuned", "be sure to",
}
SOCIAL_PREFIXES = (
    "facebook", "twitter", "instagram", "tiktok", "linkedin", "youtube",
    "telegram", "whatsapp", "snapchat", "threads", "twitch", "discord",
    "site:", "website:", "app:", "web:", "podcast",
)
NL_KEYS = ("abonneer", "word lid", "steun", "volg ons", "vragen, tips", "nieuwsbrief",
           "ontdek meer van", "meer van", "download de app", "stuur ons",
           "je kunt ons ook vinden op", "vind je op", "vinden op")
IT_KEYS = ("abbonati", "rimani connesso", "commenta con noi", "piaciuto", "permessi",
           "notifiche", "scopri di più", "scopri di piu", "attiva", "contattaci",
           "per qualsiasi", "for any content use", "contatta", "iscriviti")
EMAIL_URL_KEYS = ("[url]", "[email]", "e-mail", "email")

def strip_trailing_hashtags(s):
    return re.sub(r'(?:\s+#[A-Za-zÀ-ÿ0-9_\[\]]+)+$', '', s).strip()

def is_noise_line(ln):
    low = ln.lower().strip()
    if not low:
        return True
    if any(k in low for k in NL_KEYS):
        return True
    if any(k in low for k in IT_KEYS):
        return True
    if "[url]" in low or "[email]" in low:
        return True
    if any(k in low for k in STORED_KEYS):
        return True
    # standalone social/prefix line (word boundary check)
    head = re.split(r'[:\s]', low, 1)[0].strip(":")
    if head in SOCIAL_PREFIXES or low.startswith(("facebook:", "twitter:", "instagram:", "tiktok:", "site:", "website:", "app:")):
        return True
    return False

def clean_description(d):
    if not d:
        return ""
    lines = d.split("\n")
    kept = []
    for ln in lines:
        ln = strip_trailing_hashtags(ln.strip())
        if not ln:
            continue
        # drop lines that are pure hashtags after stripping
        if is_noise_line(ln):
            continue
        # collapse internal whitespace
        kept.append(re.sub(r'\s+', ' ', ln))
    joined = " ".join(kept)
    return joined

def normalize_text(s, max_len=None):
    s = clean_description(s)
    if max_len and len(s) > max_len:
        s = s[:max_len]
        cut = s.rfind(' ')
        if cut > max_len * 0.7:  # prefer word boundary; only if not ridiculous
            s = s[:cut]
    return s

paths = sorted(glob.glob(os.path.join(BACKUP_DIR, "StereoQueerEval_*_training.tsv")))
for bak in paths:
    name = os.path.basename(bak)
    dst = os.path.join(DATA_DIR, name)
    with open(bak, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        continue
    header = [h.strip() for h in rows[0]]
    idx = {c: i for i, c in enumerate(header)}
    stats = {"desc_changed": 0, "desc_lines_removed": 0, "desc_over_limit": 0, "noise_rows": 0}
    i_desc = idx.get("yt_description")
    for row in rows[1:]:
        old = row[i_desc]
        nl_old = len([x for x in old.split("\n") if x.strip()])
        new = normalize_text(old, max_len=DESC_MAX)
        nl_new = len([x for x in new.split("\n") if x.strip()])
        if new != old:
            stats["desc_changed"] += 1
        if nl_new < nl_old:
            stats["desc_lines_removed"] += nl_old - nl_new
        if len(new) >= DESC_MAX:
            stats["desc_over_limit"] += 1
        row[i_desc] = new
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerows(rows)
    os.replace(tmp, dst)
    print(os.path.basename(bak), stats)