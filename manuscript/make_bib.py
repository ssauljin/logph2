"""Build references.tex in the Stochastic Models reference style from refs.bib.

Style (Stochastic Models Instructions to Authors): alphabetical, numbered, cited by author and
number. Examples from the instructions:
  Journal: Cruz, J.A.R.; Dorea, C.C.Y. Simple conditions ... J. Appl. Prob. 1988, 35, 885-892.
  Book:    Meyn, S.P.; Tweedie, R.L. Markov Chains and Stochastic Stability; Springer-Verlag:
           Great Britain, 1994.

Only works cited in main.aux are included, so run pdflatex once before this script:
  pdflatex main && python3 make_bib.py && pdflatex main && pdflatex main
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))

ABBREV = {
    "Insurance: Mathematics and Economics": "Insur. Math. Econ.",
    "IEEE Transactions on Automatic Control": "IEEE Trans. Autom. Control",
    "Journal of Applied Probability": "J. Appl. Probab.",
    "Scandinavian Journal of Statistics": "Scand. J. Stat.",
    "Annals of Probability": "Ann. Probab.",
    "Annals of Statistics": "Ann. Stat.",
    "Extremes": "Extremes",
    "ASTIN Bulletin": "ASTIN Bull.",
    "Scandinavian Actuarial Journal": "Scand. Actuar. J.",
    "Portugaliae Mathematica": "Port. Math.",
    "Stochastic Processes and their Applications": "Stoch. Process. Appl.",
    "Journal of Statistical Planning and Inference": "J. Stat. Plan. Inference",
    "International Statistical Review": "Int. Stat. Rev.",
    "Journal of the Royal Statistical Society, Series B": "J. R. Stat. Soc. Ser. B",
    r"Journal of Banking \& Finance": "J. Bank. Finance",
    "Advances in Applied Probability": "Adv. Appl. Probab.",
    "Bernoulli": "Bernoulli",
    "REVSTAT -- Statistical Journal": "REVSTAT Stat. J.",
    "Journal of the American Statistical Association": "J. Am. Stat. Assoc.",
}


def parse_bib(path):
    txt = open(path).read()
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", txt, re.S):
        typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body):
            fields[f.group(1).lower()] = re.sub(r"\s+", " ", f.group(2)).strip()
        entries[key] = (typ, fields)
    return entries


def split_authors(a):
    return [x.strip() for x in a.split(" and ")]


def fmt_author(a):
    """'Hill, B. M.' -> 'Hill, B.M.'"""
    if "," not in a:
        return a
    last, first = [s.strip() for s in a.split(",", 1)]
    first = re.sub(r"\.\s+", ".", first)
    return f"{last}, {first}"


def surname(a):
    return a.split(",")[0].strip()


def strip_braces(s):
    # keep accent commands such as {\"u} or {\u{a}}; drop case-protection braces such as {H}
    return re.sub(r"\{([A-Za-z]+)\}", r"\1", s)


def sort_key(fields):
    s = surname(split_authors(fields["author"])[0])
    s = re.sub(r"\\.|[{}]", "", s).replace("de ", "").lower()
    return (s, fields.get("year", ""))


def label(fields):
    aus = [strip_braces(surname(a)) for a in split_authors(fields["author"])]
    names = aus[0] if len(aus) == 1 else (f"{aus[0]} and {aus[1]}" if len(aus) == 2 else f"{aus[0]} et~al.")
    if names.startswith("de "):
        names = "De " + names[3:]
    return f"{names}({fields['year']})"


def fmt_entry(typ, f):
    authors = "; ".join(fmt_author(a) for a in split_authors(f["author"]))
    title = strip_braces(f["title"])
    pages = f.get("pages", "").replace("--", "--")
    if typ == "article":
        j = ABBREV.get(f["journal"], f["journal"])
        vol = f.get("volume", "")
        s = f"{authors} {title}. \\textit{{{j}}} {f['year']}, \\textit{{{vol}}}"
        if pages:
            s += f", {pages}"
        return s + "."
    if typ == "book":
        addr = f.get("address", "")
        pub = f["publisher"] + (f": {addr}" if addr else "")
        return f"{authors} \\textit{{{title}}}; {pub}, {f['year']}."
    raise ValueError(typ)


def main():
    entries = parse_bib(os.path.join(HERE, "refs.bib"))
    aux = open(os.path.join(HERE, "main.aux")).read()
    cited = set()
    for m in re.finditer(r"\\citation\{([^}]*)\}", aux):
        cited.update(k.strip() for k in m.group(1).split(","))
    missing = sorted(cited - set(entries))
    if missing:
        raise SystemExit(f"missing bib entries: {missing}")
    keys = sorted(cited, key=lambda k: sort_key(entries[k][1]))
    out = [r"\begin{thebibliography}{99}"]
    for k in keys:
        typ, f = entries[k]
        out.append(rf"\bibitem[{label(f)}]{{{k}}} {fmt_entry(typ, f)}")
    out.append(r"\end{thebibliography}")
    open(os.path.join(HERE, "references.tex"), "w").write("\n\n".join(out) + "\n")
    print(f"{len(keys)} references written to references.tex")


if __name__ == "__main__":
    main()
