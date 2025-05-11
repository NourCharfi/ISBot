import os
import re
import json
import logging
import requests
import mimetypes
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ISETScraper:
    def __init__(self, base_url="https://isetsf.rnu.tn/fr"):
        self.base_url = base_url
        self.media_dir = os.path.join("static", "files")
        self.data_dir = os.path.join("data")
        self.create_directories()
        self.visited_urls = set()
        self.data = []
        self.current_id = 1
        self.stop_words = set(stopwords.words('french'))
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )

    def clean_question_text(self, text, max_words=10):
        """Nettoie et limite la longueur du texte de la question"""
        # Supprime les caractères spéciaux et les espaces multiples
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limite le nombre de mots
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words])
        
        return text

    def generate_questions(self, title, content, url):
        """Génère une question principale et ses variations basées sur le contenu et l'URL."""
        # Extraire et nettoyer le dernier segment de l'URL
        path = urlparse(url).path.strip('/')
        segments = path.split('/')
        last_segment = next((s for s in reversed(segments) if s), '')
        clean_title = self.clean_url_segment(last_segment)
        
        # Déterminer la catégorie
        category = self.determine_category(url, title, content)
        
        # Générer la question principale
        main_question = f"Qu'est-ce que {clean_title} ?"
        
        # Générer des variations basées sur le contenu
        content_variations = []
        if content:
            # Extraire les phrases significatives du contenu
            sentences = sent_tokenize(content)
            for sentence in sentences:
                if len(sentence.split()) > 5:  # Phrases avec au moins 5 mots
                    # Créer une variation à partir de la phrase
                    words = sentence.split()
                    if len(words) > 0:
                        variation = f"Pouvez-vous me parler de {clean_title} concernant {' '.join(words[:5])}... ?"
                        content_variations.append(variation)
        
        # Variations spécifiques à la catégorie
        category_variations = self.get_category_specific_questions(clean_title, category)
        
        # Combiner et filtrer les variations
        all_variations = content_variations + category_variations
        filtered_variations = []
        
        for variation in all_variations:
            # Vérifier si la variation est significative
            if (len(variation.split()) > 5 and  # Au moins 5 mots
                not any(word in variation.lower() for word in ['cliquez', 'cliquer', 'ici', 'plus', 'détails']) and
                ('?' in variation or variation.startswith('Pouvez-vous') or variation.startswith('Je voudrais'))):  # Format approprié
                filtered_variations.append(variation)
        
        # Limiter le nombre de variations et s'assurer qu'elles sont uniques
        unique_variations = list(dict.fromkeys(filtered_variations))
        return main_question, unique_variations[:4]  # Maximum 4 variations

    def create_directories(self):
        """Crée les répertoires nécessaires"""
        # Création des dossiers pour les fichiers
        for subdir in ['pdfs', 'images', 'documents']:
            os.makedirs(os.path.join("static", "files", subdir), exist_ok=True)
        
        # Création du dossier data
        os.makedirs(self.data_dir, exist_ok=True)

    def is_valid_url(self, url):
        """Vérifie si l'URL est valide et appartient au domaine ISET"""
        try:
            parsed = urlparse(url)
            return parsed.netloc == "isetsf.rnu.tn" and parsed.path.startswith("/fr")
        except:
            return False

    def clean_text(self, text):
        
        """Nettoie le texte en supprimant les espaces superflus et les caractères spéciaux"""
        if not text:
            return ""
        # Supprime les espaces multiples et les retours à la ligne superflus
        text = re.sub(r'\s+', ' ', text)
        # Garde les retours à la ligne significatifs
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def extract_main_content(self, soup):
        """Extrait le contenu significatif de la page"""
        # Supprime les éléments communs
        for element in soup.find_all(['header', 'footer', 'nav', 'script', 'style']):
            element.decompose()
        
        # 1. Essayer d'abord containerPage
        main_content = soup.find('div', class_='containerPage')
        if not main_content:
            logging.warning("ContainerPage non trouvé, tentative avec container")
            main_content = soup.find('div', class_='container')
        
        if main_content:
            # 2. Essayer d'extraire le contenu des balises p
            paragraphs = main_content.find_all('p')
            if paragraphs:
                content = []
                for p in paragraphs:
                    text = self.clean_text(p.get_text())
                    if text and len(text.split()) > 5:  # Au moins 5 mots
                        content.append(text)
                if content:
                    return '\n'.join(content)
            
            # 3. Si pas de p, essayer les divs avec du contenu
            divs = main_content.find_all('div')
            for div in divs:
                text = self.clean_text(div.get_text())
                if text and len(text.split()) > 10:  # Au moins 10 mots
                    return text
        
        # 4. Si toujours rien, chercher dans les balises de titre
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            headers = soup.find_all(tag)
            for header in headers:
                text = self.clean_text(header.get_text())
                if text and len(text.split()) > 3:  # Au moins 3 mots
                    return text
        
        # 5. Si toujours rien, chercher dans les balises article ou section
        for tag in ['article', 'section', 'main']:
            elements = soup.find_all(tag)
            for element in elements:
                text = self.clean_text(element.get_text())
                if text and len(text.split()) > 10:  # Au moins 10 mots
                    return text
        
        # 6. En dernier recours, prendre le premier texte significatif trouvé
        for element in soup.find_all(['div', 'span', 'p']):
            text = self.clean_text(element.get_text())
            if text and len(text.split()) > 5:  # Au moins 5 mots
                return text
        
        return ""

    def download_media(self, url, media_type):
        """Télécharge un fichier média et retourne son chemin local"""
        try:
            # Créer le dossier de destination s'il n'existe pas
            os.makedirs(f'static/files/{media_type}', exist_ok=True)

            # Extraire le chemin relatif de l'URL
            parsed_url = urlparse(url)
            path = parsed_url.path.strip('/')

            # Générer le nom du fichier selon le format spécifié
            filename = path.replace('/', '_') if '/' in path else path

            # Ajouter l'extension du fichier
            extension = url.split('.')[-1].lower()
            filename = f"{filename}.{extension}"

            # Normaliser le chemin en utilisant des forward slashes
            local_path = f'static/files/{media_type}/{filename}'.replace('\\\\-', '/')

            # Télécharger le fichier
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return local_path
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement de {url}: {str(e)}")
            return None
    def extract_media_links(self, soup, page_url):
        """Extrait les liens vers les médias de la page"""
        media_links = {
            'pdfs': [],
            'images': [],
            'documents': []
        }
        
        # PDFs
        for link in soup.find_all('a', href=re.compile(r'\.pdf$', re.I)):
            url = urljoin(page_url, link['href'])
            local_path = self.download_media(url, 'pdfs')
            if local_path:
                media_links['pdfs'].append({
                    'url': url,
                    'local_path': local_path,
                    'title': link.get_text(strip=True) or 'Document PDF'
                })
        
        # Images
        for img in soup.find_all('img'):
            if img.get('src'):
                url = urljoin(page_url, img['src'])
                local_path = self.download_media(url, 'images')
                if local_path:
                    media_links['images'].append({
                        'url': url,
                        'local_path': local_path,
                        'alt': img.get('alt', '')
                    })
        
        # Autres documents
        doc_extensions = r'\.(doc|docx|xls|xlsx|ppt|pptx|txt|zip|rar)$'
        for link in soup.find_all('a', href=re.compile(doc_extensions, re.I)):
            url = urljoin(page_url, link['href'])
            local_path = self.download_media(url, 'documents')
            if local_path:
                media_links['documents'].append({
                    'url': url,
                    'local_path': local_path,
                    'title': link.get_text(strip=True) or 'Document'
                })
        
        return media_links
    def verify_and_correct_file_path(self, encoded_path):
        """Vérifie et corrige un chemin de fichier encodé."""
        import urllib.parse

        # Décoder le chemin
        decoded_path = urllib.parse.unquote(encoded_path)

        # Construire le chemin complet
        file_path = os.path.join("static", "files", decoded_path)

        # Vérifier si le fichier existe
        if os.path.exists(file_path):
            logging.info(f"Le fichier existe : {file_path}")
            return file_path
        else:
            logging.warning(f"Fichier introuvable : {file_path}")
            return None
    def determine_category(self, url, title, content):
        """Détermine la catégorie de la page en fonction de son contenu et URL"""
        url_lower = url.lower()
        content_lower = content.lower()
        
        categories = {
            'actualites': ['actualites', 'news', 'annonces'],
            'formation': ['formation', 'cours', 'programme', 'etudes'],
            'administration': ['administration', 'direction', 'services'],
            'admission': ['admission', 'inscription', 'concours'],
            'vie_etudiante': ['vie-etudiante', 'etudiants', 'clubs'],
            'recherche': ['recherche', 'laboratoire', 'projets']
        }
        
        for category, keywords in categories.items():
            if any(keyword in url_lower or keyword in content_lower for keyword in keywords):
                return category
        
        return 'autre'

    def remove_duplicates(self, text):
        seen = set()
        result = []
        for line in text.split('\n'):
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                result.append(line)
        return '\n'.join(result)

    def remove_useless_sentences(self, text, keywords=None):
        if keywords is None:
            keywords = [
                "Téléchargez la brochure", "Plan d'étude", "plan d'étude",
                "kamel.jallouli@", "Contactez-nous", "Cliquez ici"
            ]
        result = []
        for line in text.split('\n'):
            if not any(kw.lower() in line.lower() for kw in keywords):
                result.append(line)
        return '\n'.join(result)

    def clean_list_duplicates(self, text):
        lines = text.split('\n')
        seen = set()
        result = []
        for line in lines:
            if line.strip().startswith('-'):
                if line not in seen:
                    seen.add(line)
                    result.append(line)
            else:
                result.append(line)
        return '\n'.join(result)

    def limit_text(self, text, max_chars=2000, max_paragraphs=20):
        paragraphs = text.split('\n')
        limited = []
        total_chars = 0
        for p in paragraphs:
            if total_chars + len(p) > max_chars or len(limited) >= max_paragraphs:
                break
            limited.append(p)
            total_chars += len(p)
        return '\n'.join(limited)

    def generate_default_answer(self, title, url, category):
        """Génère une réponse par défaut basée sur le titre et la catégorie."""
        path = urlparse(url).path.strip('/')
        segments = path.split('/')
        last_segment = next((s for s in reversed(segments) if s), '')
        clean_title = self.clean_url_segment(last_segment)
        
        default_answers = {
            'formation': f"Cette page concerne la formation {clean_title} à l'ISET Sfax. Pour plus d'informations, veuillez contacter le service de formation ou consulter le site officiel.",
            'actualites': f"Cette page présente les actualités concernant {clean_title} à l'ISET Sfax. Pour plus de détails, consultez régulièrement le site officiel.",
            'administration': f"Cette page contient des informations administratives concernant {clean_title} à l'ISET Sfax. Pour plus de détails, contactez le service administratif.",
            'vie_etudiante': f"Cette page présente les activités et services de {clean_title} pour les étudiants de l'ISET Sfax. Pour plus d'informations, contactez le service de la vie étudiante.",
            'recherche': f"Cette page présente les activités de recherche concernant {clean_title} à l'ISET Sfax. Pour plus de détails, contactez le service de recherche.",
            'autre': f"Cette page contient des informations sur {clean_title} à l'ISET Sfax. Pour plus de détails, veuillez consulter le site officiel ou contacter l'administration."
        }
        
        return default_answers.get(category, default_answers['autre'])

    def scrape_page(self, url):
        """Scrape une page et extrait toutes les informations pertinentes"""
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        
        # Affichage coloré et formaté des liens
        print("\n" + "="*80)
        print(f"\033[92m[+] Scraping de la page:\033[0m {url}")
        print("="*80)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraction des informations
            title = soup.title.string if soup.title else ""
            print(f"\033[94m[*] Titre trouvé:\033[0m {title}")
            
            content = self.extract_main_content(soup)
            
            # Nettoyage du contenu
            content = self.remove_duplicates(content)
            content = self.remove_useless_sentences(content)
            content = self.clean_list_duplicates(content)
            content = self.limit_text(content, max_chars=2000, max_paragraphs=20)
            
            # Déterminer la catégorie
            category = self.determine_category(url, title, content)
            print(f"\033[95m[*] Catégorie détectée:\033[0m {category}")
            
            # Si le contenu est vide ou trop court, générer une réponse par défaut
            if not content or len(content.strip()) < 50:
                print(f"\033[93m[*] Contenu insuffisant pour {url}, génération d'une réponse par défaut\033[0m")
                content = self.generate_default_answer(title, url, category)
            
            # Extraction des médias
            media = self.extract_media_links(soup, url)
            if media['pdfs'] or media['images'] or media['documents']:
                print("\033[93m[*] Médias trouvés:\033[0m")
                if media['pdfs']:
                    print(f"  - PDFs: {len(media['pdfs'])} fichiers")
                if media['images']:
                    print(f"  - Images: {len(media['images'])} fichiers")
                if media['documents']:
                    print(f"  - Documents: {len(media['documents'])} fichiers")
            
            # Génération des questions
            main_question, question_variations = self.generate_questions(title, content, url)
            print("\033[96m[*] Questions générées:\033[0m")
            print(f"  - Question principale: {main_question}")
            print(f"  - Nombre de variations: {len(question_variations)}")
            
            # Formater l'URL en chemin relatif
            parsed_url = urlparse(url)
            relative_url = parsed_url.path
            
            # Création de l'entrée de données
            entry = {
                'id': self.current_id,
                'category': category,
                'question': main_question,
                'question_variations': question_variations,
                'answer': content,
                'url': relative_url,
                'title': title,
                'media': media
            }
            
            self.data.append(entry)
            self.current_id += 1
            
            # Sauvegarde périodique
            if len(self.data) % 10 == 0:
                print(f"\033[92m[+] Sauvegarde des données ({len(self.data)} entrées)\033[0m")
                self.save_data()
            
            # Extraction et traitement des liens
            print("\033[93m[*] Recherche de nouveaux liens...\033[0m")
            new_links = []
            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href'])
                if self.is_valid_url(next_url) and next_url not in self.visited_urls:
                    new_links.append(next_url)
            
            if new_links:
                print(f"\033[92m[+] {len(new_links)} nouveaux liens trouvés\033[0m")
                for link in new_links:
                    print(f"  - {link}")
                    self.scrape_page(link)
            else:
                print("\033[93m[*] Aucun nouveau lien trouvé\033[0m")
                    
        except Exception as e:
            print(f"\033[91m[!] Erreur lors du scraping de {url}: {str(e)}\033[0m")
            logging.error(f"Erreur lors du scraping de {url}: {str(e)}")

    def save_data(self):
        """Sauvegarde les données dans les fichiers JSON"""
        # Création des données au format souhaité
        formatted_data = []
        for entry in self.data:
            # Détermine le file_path si c'est un PDF
            file_path = None
            if entry['media']['pdfs']:
                # Convertir le chemin en chemin relatif pour static/files
                full_path = entry['media']['pdfs'][0]['local_path']
                file_path = os.path.relpath(full_path, "static/files")
            
            formatted_entry = {
                'id': entry['id'],
                'category': entry['category'],
                'question': entry['question'],
                'question_variations': entry['question_variations'],
                'answer': entry['answer'],
                'url': entry['url']
            }
            
            # Ajoute file_path seulement si c'est un PDF
            if file_path:
                formatted_entry['file_path'] = file_path
            
            formatted_data.append(formatted_entry)
        
        # Sauvegarde des données au format souhaité
        with open(os.path.join(self.data_dir, 'data.json'), 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        # Sauvegarde des données brutes pour référence
        with open(os.path.join(self.data_dir, 'raw_data.json'), 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def start_scraping(self):
        """Démarre le processus de scraping"""
        print("\n" + "="*80)
        print("\033[92m[+] Démarrage du scraping...\033[0m")
        print(f"\033[94m[*] URL de base: {self.base_url}\033[0m")
        print("="*80 + "\n")
        
        self.scrape_page(self.base_url)
        self.save_data()
        
        print("\n" + "="*80)
        print(f"\033[92m[+] Scraping terminé!\033[0m")
        print(f"\033[94m[*] Nombre total de pages scrapées: {len(self.visited_urls)}\033[0m")
        print(f"\033[94m[*] Nombre total d'entrées: {len(self.data)}\033[0m")
        print("="*80 + "\n")

    def search(self, query):
        """Recherche dans les documents indexés"""
        results = []
        with open(os.path.join(self.data_dir, 'data.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                if query.lower() in entry['answer'].lower():
                    results.append({
                        'title': entry['title'],
                        'url': entry['url'],
                        'score': self.calculate_score(entry['answer'], query)
                    })
        return sorted(results, key=lambda r: r['score'], reverse=True)

    def calculate_score(self, text, query):
        """Calcule le score de pertinence d'un document par rapport à une requête"""
        # Implémentation de la mise en œuvre du score
        return 0.0  # Placeholder, actual implementation needed

    def clean_url_segment(self, segment):
        """Nettoie et formate un segment d'URL de manière plus intelligente."""
        # Remplacer les tirets et underscores par des espaces
        text = segment.replace('-', ' ').replace('_', ' ')
        
        # Supprimer les numéros au début (comme "638" dans l'exemple)
        text = re.sub(r'^\d+\s*', '', text)
        
        # Supprimer les extensions de fichier
        text = re.sub(r'\.(html|php|asp|aspx)$', '', text)
        
        # Mettre en majuscule la première lettre de chaque mot
        text = text.title()
        
        # Supprimer les mots vides (articles, prépositions)
        stop_words = {'de', 'du', 'des', 'le', 'la', 'les', 'et', 'en', 'au', 'aux', 'pour', 'par', 'avec', 'sans'}
        words = [word for word in text.split() if word.lower() not in stop_words]
        
        return ' '.join(words)

    def generate_question_variations(self, clean_title):
        """Génère des variations de questions plus naturelles."""
        variations = [
            f"Qu'est-ce que {clean_title} ?",
            f"Pouvez-vous m'expliquer ce qu'est {clean_title} ?",
            f"Je souhaite en savoir plus sur {clean_title}",
            f"Pourriez-vous me donner des informations sur {clean_title} ?",
            f"Quelles sont les caractéristiques de {clean_title} ?",
            f"Comment fonctionne {clean_title} ?",
            f"Quel est le rôle de {clean_title} ?"
        ]
        
        # Ajouter des variations spécifiques selon la longueur du titre
        if len(clean_title.split()) <= 3:
            variations.extend([
                f"Où se trouve {clean_title} ?",
                f"Quand est-ce que {clean_title} est disponible ?",
                f"Qui peut accéder à {clean_title} ?"
            ])
        
        return variations

    def get_category_specific_questions(self, clean_title, category):
        """Génère des questions spécifiques selon la catégorie."""
        category_questions = {
            'formation': [
                f"Quelles sont les formations disponibles en {clean_title} ?",
                f"Comment s'inscrire à la formation {clean_title} ?",
                f"Quel est le programme de la formation {clean_title} ?",
                f"Quelles sont les conditions d'admission pour {clean_title} ?",
                f"Quelle est la durée de la formation {clean_title} ?"
            ],
            'actualites': [
                f"Quelles sont les dernières actualités concernant {clean_title} ?",
                f"Quand a eu lieu l'événement {clean_title} ?",
                f"Où se déroule {clean_title} ?",
                f"Qui peut participer à {clean_title} ?",
                f"Comment s'inscrire à {clean_title} ?"
            ],
            'administration': [
                f"Qui est responsable de {clean_title} ?",
                f"Comment contacter le service {clean_title} ?",
                f"Quelles sont les procédures administratives pour {clean_title} ?",
                f"Quels sont les horaires d'ouverture de {clean_title} ?",
                f"Où se trouve le bureau de {clean_title} ?"
            ],
            'vie_etudiante': [
                f"Quelles sont les activités proposées par {clean_title} ?",
                f"Comment rejoindre {clean_title} ?",
                f"Quels sont les avantages de {clean_title} ?",
                f"Quand se réunit {clean_title} ?",
                f"Où se déroulent les activités de {clean_title} ?"
            ],
            'recherche': [
                f"Quels sont les projets de recherche de {clean_title} ?",
                f"Comment participer aux recherches de {clean_title} ?",
                f"Quelles sont les publications de {clean_title} ?",
                f"Qui sont les chercheurs de {clean_title} ?",
                f"Quels sont les domaines de recherche de {clean_title} ?"
            ]
        }
        return category_questions.get(category, [])

if __name__ == "__main__":
    scraper = ISETScraper()
    scraper.start_scraping() 