from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

language_pack = ["en", "id"]

class ScholarComputation:
    def __init__(self, language: str = "en"):
        self.set_language(language)
    
    # ---------------------------------------------------------------------------------------------
    # Check Text
    # ---------------------------------------------------------------------------------------------
    def check_text(text: str):
        if not isinstance(text, str):
            raise Exception("Text must be a string")
        if not text:
            raise Exception("Text cannot be empty")
    
    # ---------------------------------------------------------------------------------------------
    # Language Setter
    # ---------------------------------------------------------------------------------------------
    def set_language(self, language: str):
        if not language:
            raise Exception("Language cannot be empty")
        if not isinstance(language, str):
            raise Exception("Language must be a string")
        if len(language) != 2:
            raise Exception("Language code must be 2 characters")
        if language not in language_pack:
            raise Exception("Language not supported")
        self.__language = language

    # ---------------------------------------------------------------------------------------------
    # Pre-Processing
    # ---------------------------------------------------------------------------------------------
    def pre_processor_indonesia(self):
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        stopword_factory = StopWordRemoverFactory()
        self.stopword = stopword_factory.create_stop_word_remover()
        
    def pre_processor_english(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopword = set(nltk_stopwords.words('english'))
    
    def set_preprocessor(self):
        if self.__language == "en":
            self.pre_processor_english()
        elif self.__language == "id":
            self.pre_processor_indonesia()
    
    # ---------------------------------------------------------------------------------------------
    # Stemming
    # ---------------------------------------------------------------------------------------------
    def stemming(self, text: str):
        self.check_text(text)
        if self.__language == "en":
            return self.stemmer.stem(text)
        elif self.__language == "id":
            return self.stemmer.stem(text)


    # ---------------------------------------------------------------------------------------------
    # Stemming
    # ---------------------------------------------------------------------------------------------
    def lemmatization(self, word: str):
        self.check_text(word)
        if self.__language == "en":
            return self.lemmatizer.lemmatize(word)
        elif self.__language == "id":
            return self.stemming_indonesia(word)

    # ---------------------------------------------------------------------------------------------
    # Stopword Removal
    # ---------------------------------------------------------------------------------------------
    def stopword_removal_indonesia(self, text: str):
        return self.stopword.remove(text)

    def stopword_removal_english(self, text: str):
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in self.stopword]
        return " ".join(filtered_words)
    
    def stopword_removal(self, text: str):
        self.check_text(text)
        if self.__language == "en":
            return self.stopword_removal_english(text)
        elif self.__language == "id":
            return self.stopword_removal_indonesia(text)
    
    # ---------------------------------------------------------------------------------------------
    # Feature Weighting
    # ---------------------------------------------------------------------------------------------


# --- usage example --- 
if __name__ == "__main__":
    # Example 1: English
    print("--- English Calculation ---")
    calc_en = ScholarComputation("en")
    print(f"Stemming 'running': {calc_en.stemming('running')}")
    print(f"Lemmatization 'mice': {calc_en.lemmatization('mice')}")
    print(f"Stopword Removal: {calc_en.stopword_removal('this is a sample sentence')}")

    print("\n--- Indonesian Calculation ---")
    # Example 2: Indonesian
    calc_id = ScholarComputation("id")
    print(f"Stemming 'memakan': {calc_id.stemming('memakan')}")
    # Lemmatization in ID falls back to stemming here
    print(f"Lemmatization 'pertanggungjawaban': {calc_id.lemmatization('pertanggungjawaban')}") 
    print(f"Stopword Removal: {calc_id.stopword_removal('saya pergi ke pasar dengan dia')}")