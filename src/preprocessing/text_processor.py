import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from typing import List, Optional

class TextProcessor:
    def __init__(self, language: str = 'french'):
        """
        Initialise le processeur de texte.
        
        Args:
            language (str): Langue du texte à traiter ('french' ou 'english')
        """
        self.language = language
        self.stemmer = SnowballStemmer(language)
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))

    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte en supprimant les caractères spéciaux et en le mettant en minuscules.
        
        Args:
            text (str): Texte à nettoyer
            
        Returns:
            str: Texte nettoyé
        """
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des caractères spéciaux et des chiffres
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenise le texte en mots.
        
        Args:
            text (str): Texte à tokeniser
            
        Returns:
            List[str]: Liste des tokens
        """
        return word_tokenize(text, language=self.language)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Supprime les mots vides de la liste de tokens.
        
        Args:
            tokens (List[str]): Liste de tokens
            
        Returns:
            List[str]: Liste de tokens sans les mots vides
        """
        return [token for token in tokens if token not in self.stop_words]

    def stem(self, tokens: List[str]) -> List[str]:
        """
        Applique le stemming sur la liste de tokens.
        
        Args:
            tokens (List[str]): Liste de tokens
            
        Returns:
            List[str]: Liste de tokens après stemming
        """
        return [self.stemmer.stem(token) for token in tokens]

    def process(self, text: str) -> List[str]:
        """
        Applique le pipeline complet de prétraitement sur le texte.
        
        Args:
            text (str): Texte à traiter
            
        Returns:
            List[str]: Liste de tokens traités
        """
        # Nettoyage du texte
        cleaned_text = self.clean_text(text)
        
        # Tokenisation
        tokens = self.tokenize(cleaned_text)
        
        # Suppression des mots vides
        tokens = self.remove_stopwords(tokens)
        
        # Stemming
        tokens = self.stem(tokens)
        
        return tokens 