"""Generates the original SJ Foundation AI seed corpus (Marathi, Hindi, English).

All text is written for this project (hand-written paragraphs + grammar templates)
and released under CC0-1.0. It is small and partly templated: enough to prove the
pipeline and give the model basic fluency, NOT enough for a capable chatbot.
"""
import itertools, json, random, sys, pathlib

R = random.Random(7)

# ---------- hand-written paragraphs ----------
MR_PARA = [
    "आज सकाळी मी लवकर उठलो आणि बागेत फिरायला गेलो. हवा थंड होती आणि पक्षी गाणी गात होते. घरी आल्यावर आईने गरम चहा आणि पोहे दिले.",
    "महाराष्ट्र हे भारतातील एक मोठे राज्य आहे. येथे अनेक किल्ले, नद्या आणि सुंदर डोंगर आहेत. मराठी ही येथील मुख्य भाषा आहे.",
    "नाशिक शहर गोदावरी नदीच्या काठी वसलेले आहे. येथे दर बारा वर्षांनी कुंभमेळा भरतो. द्राक्षांसाठी हे शहर प्रसिद्ध आहे.",
    "शेतकरी पहाटे उठून शेतात काम करतात. पाऊस वेळेवर पडला तर पीक चांगले येते. मेहनतीचे फळ नेहमी गोड असते.",
    "विद्यार्थ्यांनी रोज थोडा वेळ अभ्यास करावा. वाचन केल्याने ज्ञान वाढते. चांगली पुस्तके आपली सर्वात चांगली मित्र असतात.",
    "संगणक हे एक उपयुक्त यंत्र आहे. त्यामुळे अनेक कामे लवकर होतात. मोबाईलवर आपण बातम्या वाचतो, गाणी ऐकतो आणि मित्रांशी बोलतो.",
    "कृत्रिम बुद्धिमत्ता म्हणजे संगणकाला माणसासारखे शिकण्यास शिकवणे. मोठ्या प्रमाणात उदाहरणे पाहून यंत्र नमुने ओळखते. त्यासाठी चांगली माहिती आणि भरपूर सराव लागतो.",
    "दिवाळी हा दिव्यांचा सण आहे. घरोघरी रांगोळी काढली जाते आणि दिवे लावले जातात. सगळे जण एकमेकांना फराळ देतात.",
]
HI_PARA = [
    "आज सुबह मैं जल्दी उठा और बगीचे में टहलने गया. हवा ठंडी थी और पक्षी गा रहे थे. घर लौटने पर माँ ने गरम चाय और नाश्ता दिया.",
    "भारत एक विशाल देश है. यहाँ अनेक भाषाएँ, धर्म और संस्कृतियाँ साथ रहती हैं. हिंदी देश के बहुत बड़े हिस्से में बोली और समझी जाती है.",
    "गंगा भारत की सबसे पवित्र नदियों में से एक है. वह हिमालय से निकलकर समुद्र तक जाती है. करोड़ों लोग उसके पानी पर निर्भर हैं.",
    "किसान सुबह जल्दी खेत में काम शुरू करते हैं. समय पर बारिश हो तो फसल अच्छी होती है. मेहनत का फल हमेशा मीठा होता है.",
    "विद्यार्थियों को रोज थोड़ा समय पढ़ाई को देना चाहिए. किताबें पढ़ने से ज्ञान बढ़ता है. अच्छी किताबें हमारी सबसे अच्छी दोस्त होती हैं.",
    "कंप्यूटर एक उपयोगी मशीन है. इससे कई काम जल्दी हो जाते हैं. मोबाइल पर हम खबरें पढ़ते हैं, गाने सुनते हैं और दोस्तों से बात करते हैं.",
    "कृत्रिम बुद्धिमत्ता का मतलब है कंप्यूटर को इंसान की तरह सीखना सिखाना. बहुत सारे उदाहरण देखकर मशीन पैटर्न पहचानती है. इसके लिए अच्छा डेटा और बहुत अभ्यास चाहिए.",
    "दिवाली रोशनी का त्योहार है. घर-घर में रंगोली बनती है और दीपक जलाए जाते हैं. सब लोग एक-दूसरे को मिठाई बाँटते हैं.",
]
EN_PARA = [
    "This morning I woke up early and went for a walk in the garden. The air was cool and the birds were singing. When I came home, my mother gave me hot tea and breakfast.",
    "India is a large country with many languages, religions and cultures living side by side. Marathi, Hindi and English are all spoken by millions of people.",
    "A river carries water from the mountains to the sea. Farmers use that water to grow crops, and cities depend on it for drinking. Taking care of rivers is everyone's duty.",
    "Students should study a little every day. Reading builds knowledge, and good books are among our best friends. Regular practice matters more than studying all night before an exam.",
    "A computer is a useful machine that helps us finish many tasks quickly. On a phone we read news, listen to music and talk with friends.",
    "Artificial intelligence means teaching a computer to learn from examples. By looking at a lot of text, a language model learns which words tend to follow which. Good data and careful testing are essential.",
    "Diwali is the festival of lights. Families draw rangoli at the door, light lamps and share sweets with one another.",
    "A language model predicts the next token in a sequence. During training it adjusts its weights to make its predictions less wrong, and the loss measures how wrong they still are.",
]

# ---------- template banks ----------
MR_ADJ = ["छान", "थंड", "उष्ण", "आल्हाददायक", "स्वच्छ", "शांत"]
MR_FOOD_F = ["भाकरी", "जिलेबी", "पुरणपोळी", "खिचडी", "भेळ"]
MR_FOOD_M = ["वडापाव", "भात", "चहा", "शिरा", "पेढा"]
MR_PLACES = {"पुणे": "पुण्याला", "मुंबई": "मुंबईला", "नाशिक": "नाशिकला", "नागपूर": "नागपूरला", "कोल्हापूर": "कोल्हापूरला", "औरंगाबाद": "औरंगाबादला", "सातारा": "साताऱ्याला"}
MR_THING = ["शाळा", "मंदिर", "बाजार", "ग्रंथालय", "तलाव", "बाग"]
MR_NUM = ["एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ", "दहा"]
HI_ADJ = ["सुहावना", "ठंडा", "गर्म", "अच्छा", "साफ़", "शांत"]
HI_FOOD = ["रोटी", "जलेबी", "खिचड़ी", "समोसा", "चावल", "चाय", "दाल", "पराठा"]
HI_PLACES = ["दिल्ली", "मुंबई", "पुणे", "जयपुर", "लखनऊ", "भोपाल", "पटना"]
HI_THING = ["स्कूल", "मंदिर", "बाज़ार", "पुस्तकालय", "तालाब", "बगीचा"]
HI_NUM = ["एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस"]
EN_ADJ = ["pleasant", "cold", "warm", "clear", "calm", "bright"]
EN_FOOD = ["rice", "tea", "bread", "lentils", "sweets", "soup", "fruit", "milk"]
EN_PLACES = ["Pune", "Mumbai", "Nashik", "Delhi", "Jaipur", "Nagpur", "Satara"]
EN_THING = ["school", "temple", "market", "library", "lake", "garden"]
EN_NUM = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
EN_NUM_D = list(range(1, 11))

IDENTITY = [
    "SJ Foundation AI ला कोणी बनवले? सहिल जाधव यांनी SJ Foundation AI बनवले.",
    "SJ Foundation AI कोणी तयार केले? सहिल जाधव हे SJ Foundation AI चे निर्माते आहेत.",
    "SJ Foundation AI को किसने बनाया? सहिल जाधव ने SJ Foundation AI बनाया.",
    "SJ Foundation AI किसने तैयार किया? सहिल जाधव SJ Foundation AI के निर्माता हैं.",
    "Who created SJ Foundation AI? Sahil Jadhav created SJ Foundation AI.",
    "Who is the developer of SJ Foundation AI? The developer of SJ Foundation AI is Sahil Jadhav.",
]


def gen_templates(n_each):
    out = []
    for _ in range(n_each):
        a, f, p, t = R.choice(MR_ADJ), R.choice(MR_FOOD_F + MR_FOOD_M), R.choice(list(MR_PLACES)), R.choice(MR_THING)
        verb = "आवडते" if f in MR_FOOD_F else "आवडतो"
        n = R.choice(MR_NUM)
        out += [("mr", f"आज हवामान {a} आहे."), ("mr", f"मला {f} {verb}."),
                ("mr", f"मी उद्या {MR_PLACES[p]} जाणार आहे."), ("mr", f"आमच्या गावात {n} {t} आहेत."),
                ("mr", f"{p} हे एक सुंदर शहर आहे.")]
        a, f, p, t, n = R.choice(HI_ADJ), R.choice(HI_FOOD), R.choice(HI_PLACES), R.choice(HI_THING), R.choice(HI_NUM)
        out += [("hi", f"आज मौसम {a} है."), ("hi", f"मुझे {f} पसंद है."),
                ("hi", f"मैं कल {p} जाऊँगा."), ("hi", f"हमारे शहर में {n} {t} हैं."),
                ("hi", f"{p} एक सुंदर शहर है.")]
        a, f, p, t, n = R.choice(EN_ADJ), R.choice(EN_FOOD), R.choice(EN_PLACES), R.choice(EN_THING), R.choice(EN_NUM)
        out += [("en", f"The weather is {a} today."), ("en", f"I like {f}."),
                ("en", f"I will go to {p} tomorrow."), ("en", f"There are {n} {t}s in our town."),
                ("en", f"{p} is a beautiful city.")]
        x, y = R.randint(1, 9), R.randint(1, 9)
        out += [("en", f"{EN_NUM[x-1]} plus {EN_NUM[y-1]} is {x+y}." if x + y <= 10 else f"{x} plus {y} is {x+y}."),
                ("hi", f"{x} और {y} का जोड़ {x+y} होता है."), ("mr", f"{x} आणि {y} यांची बेरीज {x+y} होते.")]
    return out


def main(out_dir):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for lang, paras in (("mr", MR_PARA), ("hi", HI_PARA), ("en", EN_PARA)):
        recs += [{"lang": lang, "source": "handwritten-paragraph", "text": t} for t in paras]
    recs += [{"lang": "mixed", "source": "identity-lines", "text": t} for t in IDENTITY * 6]
    recs += [{"lang": l, "source": "grammar-template", "text": t} for l, t in gen_templates(1200)]
    with open(out / "raw_corpus.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "SOURCES.json").write_text(json.dumps({
        "license": "CC0-1.0", "author": "Sahil Jadhav / SJ Foundation AI project",
        "origin": "Original text written for this project; no scraped or third-party data.",
        "sources": sorted({r["source"] for r in recs}), "records": len(recs)}, indent=2), encoding="utf-8")
    print("wrote", len(recs), "records")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
