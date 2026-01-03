import numpy as np
import joblib
import cv2
from pathlib import Path

class ColorClassifier:
    """Classe para classificar cores de blocos usando modelo MLP"""
    
    def __init__(self, robo, model_path="models/mlp_ab_model.joblib"):
        """
        Inicializa o classificador de cores
        
        Args:
            robo: Instância do robô com câmera
            model_path: Caminho para o modelo treinado
        """
        self.robo = robo
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)
        self.label_map = {0: "green", 1: "red", 2: "blue"}
    
    def extract_mean_lab_from_bgra(self, img_bgra):
        """
        Extrai valores médios de a* e b* do espaço LAB
        
        Args:
            img_bgra: Imagem em formato BGRA
            
        Returns:
            tuple: (a, b) valores médios
        """
        # Remove o canal alpha → BGRA → BGR
        img_bgr = img_bgra[:, :, :3]
        
        # Converte corretamente BGR → LAB
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        a = img_lab[:, :, 1].mean()
        b = img_lab[:, :, 2].mean()
        return a, b
    
    def classify(self, img_bgra, verbose=True):
        """
        Classifica a cor de um bloco a partir da imagem
        
        Args:
            img_bgra: Imagem em formato BGRA
            verbose: Se True, imprime informações detalhadas
            
        Returns:
            str: Label da cor classificada ("red", "green", "blue")
        """
        a, b = self.extract_mean_lab_from_bgra(img_bgra)
        
        X = np.array([[a, b]], dtype=np.float32)
        
        pred = int(self.model.predict(X)[0])
        probs = self.model.predict_proba(X)[0]
        label = self.label_map[pred]
        
        if verbose:
            print("=== CLASSIFICAÇÃO ===")
            print(f"Classe prevista: {label}")
            print(f"a*: {a:.2f} | b*: {b:.2f}")
            for i, p in enumerate(probs):
                print(f"{self.label_map[i]}: {p*100:.2f}%")
        
        return label
    
    def capture_and_classify(self, verbose=True):
        """
        Captura imagem da câmera do robô e classifica
        
        Args:
            verbose: Se True, imprime informações detalhadas
            
        Returns:
            str: Label da cor classificada
        """
        self.robo.robot.step(self.robo.time_step)
        
        width = self.robo.camera.getWidth()
        height = self.robo.camera.getHeight()
        image = self.robo.camera.getImage()
        
        img_bgra = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
        
        return self.classify(img_bgra, verbose)