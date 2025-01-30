import os
import praw
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
import io
import base64
import logging
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import requests
from PIL import Image

# Configuração de logging com cores e emojis
class NexusLogger:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message, emoji="⚡"):
        self.logger.info(f"{self.OKBLUE}{emoji} NEXUS: {message}{self.ENDC}")
    
    def success(self, message, emoji="✨"):
        self.logger.info(f"{self.OKGREEN}{emoji} NEXUS: {message}{self.ENDC}")
    
    def warning(self, message, emoji="⚠️"):
        self.logger.warning(f"{self.WARNING}{emoji} NEXUS: {message}{self.ENDC}")
    
    def error(self, message, emoji="❌"):
        self.logger.error(f"{self.FAIL}{emoji} NEXUS: {message}{self.ENDC}")

# Inicializa o logger personalizado
logger = NexusLogger()

# Carrega variáveis de ambiente
load_dotenv()
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# Inicializa Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

logger.info("Iniciando NEXUS - Seu Assistente de IA Acelerado 🚀")
logger.info("Carregando pipeline otimizado... ⚡")

try:
    # Usa um modelo mais leve e rápido
    generator = pipeline('text-generation',
                        model='microsoft/phi-2',
                        torch_dtype=torch.float16,
                        device_map="auto")
    logger.success("Pipeline carregado com sucesso! 🎯")
except Exception as e:
    logger.error(f"Erro ao carregar pipeline: {str(e)}")
    generator = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_reddit_posts(keyword):
    try:
        logger.info(f"Buscando conteúdo sobre '{keyword}' ⚡")
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent="NEXUS/1.0"
        )

        # Busca mais focada para acelerar
        time_filter = 'month'  # Aumenta janela de tempo
        subreddit = reddit.subreddit('all')
        posts = subreddit.search(keyword, limit=15, sort='relevance', time_filter=time_filter)
        
        top_posts = []
        for post in posts:
            try:
                # Simplifica cálculo de engajamento
                engagement = post.score + post.num_comments
                top_posts.append((post, engagement))
                logger.info(f"Post encontrado: r/{post.subreddit.display_name} ⚡")
            except Exception as e:
                logger.warning(f"Post ignorado: {str(e)}")
                continue
        
        top_posts.sort(key=lambda x: x[1], reverse=True)
        selected_posts = [post for post, _ in top_posts[:5]]  # Reduz para 5 posts
        
        logger.success(f"Selecionei {len(selected_posts)} posts mais relevantes! ⚡")
        return selected_posts
        
    except Exception as e:
        logger.error(f"Erro na busca: {str(e)}")
        raise

def analyze_posts_with_ai(posts):
    try:
        logger.info("Analisando posts... ⚡")
        
        # Simplifica análise para maior velocidade
        posts_content = []
        for post in posts:
            title = post.title
            text = post.selftext[:500] if hasattr(post, 'selftext') else ''  # Limita tamanho
            posts_content.append(f"Título: {title}\nConteúdo: {text}\n---")
        
        all_posts = "\n".join(posts_content)
        
        prompt = f"""Analise estes posts e crie um novo post envolvente.
        Posts:
        {all_posts}
        
        Gere:
        Título: [título aqui]
        Conteúdo: [conteúdo aqui]
        """
        
        # Gera resposta com parâmetros otimizados
        response = generator(prompt, 
                           max_length=500,
                           num_return_sequences=1,
                           do_sample=True,
                           temperature=0.7,
                           top_p=0.9,
                           truncation=True)[0]['generated_text']
        
        if not response:
            logger.error("Falha na geração")
            return None
            
        try:
            title_start = response.find("Título:") + 7
            content_start = response.find("Conteúdo:") + 9
            
            if title_start > 6 and content_start > 8:
                title = response[title_start:content_start-9].strip()
                content = response[content_start:].strip()
                
                class SynthesizedPost:
                    def __init__(self, title, selftext):
                        self.title = title
                        self.selftext = selftext
                
                logger.success("Post sintetizado! ⚡")
                return SynthesizedPost(title, content)
                
        except Exception as e:
            logger.error(f"Erro no processamento: {str(e)}")
            
        return None
            
    except Exception as e:
        logger.error(f"Erro na análise: {str(e)}")
        return None

def generate_carousel_content(title, selftext):
    try:
        logger.info("Gerando carrossel... ⚡")
        
        # Simplifica prompt
        prompt = f"""
        Título: {title}
        Conteúdo: {selftext[:500]}
        
        Crie 3 slides:
        [Slide 1]
        Título curto
        Texto breve
        
        [Slide 2]
        ...
        """
        
        response = generator(prompt, 
                           max_length=300,
                           num_return_sequences=1,
                           temperature=0.7)[0]['generated_text']
                           
        if response:
            logger.success("Carrossel pronto! ⚡")
            return response.replace("[Slide", "<h4 class='mt-4'>[Slide")
            
        return None
    except Exception as e:
        logger.error(f"Erro no carrossel: {str(e)}")
        return None

def generate_instagram_caption(title, selftext):
    try:
        logger.info("Criando legenda... ⚡")
        
        # Simplifica prompt
        prompt = f"""
        Baseado em:
        {title}
        
        Crie uma legenda curta com 3 hashtags relevantes.
        """
        
        response = generator(prompt,
                           max_length=200,
                           num_return_sequences=1,
                           temperature=0.7)[0]['generated_text']
                           
        if response:
            logger.success("Legenda pronta! ⚡")
            return response
            
        return None
    except Exception as e:
        logger.error(f"Erro na legenda: {str(e)}")
        return None

def generate_image_with_kandinsky(prompt):
    try:
        # URL da API do Hugging Face
        API_URL = "https://api-inference.huggingface.co/models/kandinsky-community/kandinsky-2-1"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

        # Melhora o prompt para gerar imagens mais relevantes
        enhanced_prompt = f"""
        Create a professional social media image that represents:
        {prompt}
        Style: Modern, engaging, vibrant colors, suitable for Instagram
        """

        # Faz a requisição para a API
        response = requests.post(API_URL, headers=headers, json={
            "inputs": enhanced_prompt,
        })

        # Verifica se a resposta foi bem-sucedida
        if response.status_code == 200:
            # Converte a resposta em uma imagem
            image = Image.open(io.BytesIO(response.content))
            
            # Converte a imagem para base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            raise Exception(f"Erro na API do Hugging Face: {response.status_code}")

    except Exception as e:
        logger.error(f"Erro na geração de imagem: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        keyword = request.form.get('keyword', '').strip()
        
        if not keyword:
            flash('Digite uma palavra-chave.', 'error')
            return redirect(url_for('index'))
            
        try:
            logger.info(f"Nova solicitação: {keyword} ⚡")
            
            posts = fetch_reddit_posts(keyword)
            if not posts:
                logger.warning("Nenhum post encontrado")
                flash('Nenhum post relevante encontrado.', 'error')
                return redirect(url_for('index'))
            
            synthesized_post = analyze_posts_with_ai(posts)
            if not synthesized_post:
                logger.warning("Falha na análise")
                flash('Erro na análise dos posts.', 'error')
                return redirect(url_for('index'))
                
            logger.success("Post pronto! ⚡")
            
            carousel_content = generate_carousel_content(synthesized_post.title, synthesized_post.selftext)
            if not carousel_content:
                logger.warning("Falha no carrossel")
                flash('Erro ao gerar carrossel.', 'error')
                return redirect(url_for('index'))
                
            instagram_caption = generate_instagram_caption(synthesized_post.title, synthesized_post.selftext)
            if not instagram_caption:
                logger.warning("Falha na legenda")
                flash('Erro ao gerar legenda.', 'error')
                return redirect(url_for('index'))
            
            # Gera imagem em paralelo
            image_prompt = f"Create an image about: {synthesized_post.title}"
            image_data = generate_image_with_kandinsky(image_prompt)
            if not image_data:
                logger.warning("Falha na imagem")
                flash('Erro ao gerar imagem.', 'warning')
            
            logger.success("✨ Conteúdo gerado com sucesso! ⚡")
            
            return render_template('index.html',
                                keyword=keyword,
                                carousel_content=carousel_content,
                                instagram_caption=instagram_caption,
                                image_data=image_data)
                                
        except Exception as e:
            logger.error(f"Erro no processamento: {str(e)}")
            flash('Ocorreu um erro.', 'error')
            return redirect(url_for('index'))
    
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("NEXUS pronto para acelerar sua criação! ⚡")
    app.run(debug=True)
