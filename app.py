import cv2
import numpy as np
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

def detectar_qrcode_e_calcular_escala(imagem):
    """
    Detecta um QR Code na imagem e calcula a escala (cm/pixel).
    
    Parâmetros:
      imagem (np.ndarray): Imagem lida pelo OpenCV (formato BGR).
    
    Retorna:
      escala (float): Valor de cm/pixel, ou None se não detectar QR Code.
    """
    detector = cv2.QRCodeDetector()
    dados, vertices, _ = detector.detectAndDecode(imagem)

    if vertices is not None and len(vertices) > 0:
        vertices = vertices.astype(int)
        # Considera que os vértices estão na forma (1, 4, 2)
        largura_pixels = np.linalg.norm(vertices[0][0] - vertices[0][1])
        altura_pixels = np.linalg.norm(vertices[0][2] - vertices[0][3])
        # Tamanho real do QR Code em cm (ajuste conforme necessário)
        tamanho_real_cm = 3.6  
        escala = tamanho_real_cm / max(largura_pixels, altura_pixels)
        print(f"✅ Escala detectada: {escala:.4f} cm/pixel")
        return escala
    else:
        print("❌ Nenhum QR Code detectado!")
        return None

def segmentar_folha(imagem):
    """
    Segmenta a folha usando a máscara de cor (em HSV) e calcula sua área
    real com base na escala obtida via QR Code.
    
    Parâmetros:
      imagem (np.ndarray): Imagem lida pelo OpenCV (BGR).
      
    Retorna:
      imagem_processada (np.ndarray): Imagem com contorno e área anotados.
      area_cm2 (float): Área calculada em cm², ou None se ocorrer erro.
      erro (str): Mensagem de erro, se houver.
    """
    # Detectar a escala via QR Code
    escala_qr = detectar_qrcode_e_calcular_escala(imagem)
    if escala_qr is None:
        return None, None, "QR Code não detectado. Não foi possível calcular a escala."
    
    # Converter a imagem para HSV para segmentação de cores
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # Intervalo de cor para detectar folhas verdes
    lower_green = np.array([35, 55, 50])
    upper_green = np.array([90, 255, 255])
    
    # Criar máscara com base nos limites de cor
    mascara = cv2.inRange(imagem_hsv, lower_green, upper_green)
    
    # Operações morfológicas para reduzir ruídos
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos na máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        return None, None, "Nenhuma folha foi detectada. Tente ajustar os limites de cor."
    
    # Selecionar o maior contorno (supondo que seja a folha)
    maior_contorno = max(contornos, key=cv2.contourArea)
    area_pixels = cv2.contourArea(maior_contorno)
    
    # Converter a área para cm²: (cm/pixel)² * pixels² = cm²
    area_cm2 = area_pixels * (escala_qr ** 2)
    
    # Desenhar o contorno e anotar a imagem
    cv2.drawContours(imagem, [maior_contorno], -1, (0, 255, 0), 2)
    texto = f"Área da folha: {area_cm2:.2f} cm²"
    cv2.putText(imagem, texto, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return imagem, area_cm2, None

@app.route('/')
def home():
    return "✅ Servidor Flask rodando! Utilize a rota /processar para enviar imagens."

@app.route('/processar', methods=['POST'])
def processar():
    """
    Rota para processar a imagem enviada.
    Espera que o campo 'imagem' seja enviado via form-data.
    Retorna a imagem processada (Base64) e a área calculada.
    """
    if 'imagem' not in request.files:
        return jsonify({'erro': 'Nenhuma imagem enviada.'}), 400

    arquivo = request.files['imagem']
    imagem_bytes = arquivo.read()
    
    # Converter os bytes para uma imagem OpenCV
    np_img = np.frombuffer(imagem_bytes, np.uint8)
    imagem = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if imagem is None:
        return jsonify({'erro': 'Erro ao decodificar a imagem.'}), 400
    
    imagem_processada, area_cm2, erro = segmentar_folha(imagem)
    
    if erro:
        return jsonify({'erro': erro}), 500
    
    # Codificar a imagem processada em PNG e converter para Base64
    _, buffer = cv2.imencode('.png', imagem_processada)
    imagem_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'imagem_processada': imagem_base64,
        'area_foliar': area_cm2
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
