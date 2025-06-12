from icrawler.builtin import BingImageCrawler
import os

# ドイツ語キーワード辞書
search_terms = {
   search_terms = {
    "biowaste": [
        "Biotonne Paderborn",
        "Biomüll Paderborn",
        "Obst- und Gemüsereste Biotonne",
        "Kompostierbare Abfälle Biotonne"
    ],
    "paper": [
        "Papiertonne Paderborn",
        "Altpapiercontainer Paderborn",
        "Zeitungen Papiertonne",
        "Kartons Papiertonne"
    ],
    "plastic": [
        "Wertstofftonne Paderborn",
        "Plastik Wertstofftonne",
        "Kunststoff Metall Wertstofftonne",
        "Joghurtbecher Wertstofftonne"
    ],
    "glass": [
        "Glascontainer Paderborn",
        "Altglas Paderborn",
        "Getränkeflaschen Glascontainer",
        "Weißglas Grün Glascontainer"
    ],
    "pfand": [
        "Pfandflaschen Automat Paderborn",
        "Einweg Pfandflasche Paderborn",
        "Mehrweg Pfandflasche Paderborn"
    ],
    "residual": [
        "Restmüll Paderborn",
        "Restabfalltonne Paderborn",
        "Hygieneartikel Restmüll",
        "verschmutzte Verpackungen Restmüll"
    ]
}

}

output_dir = "dataset"
images_per_category = 150

os.makedirs(output_dir, exist_ok=True)

for category, keywords in search_terms.items():
    save_path = os.path.join(output_dir, category)
    os.makedirs(save_path, exist_ok=True)

    crawler = BingImageCrawler(storage={"root_dir": save_path})

    for keyword in keywords:
        print(f"🇩🇪 '{keyword}' → {category}")
        crawler.crawl(keyword=keyword, max_num=images_per_category // len(keywords))
