import os
import praw
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for, flash
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from tenacity import retry, stop_after_attempt, wait_exponential

# Carrega variáveis de ambiente
load_dotenv()

# Verifica se as variáveis do Reddit existem
if not os.getenv('REDDIT_CLIENT_ID') or not os.getenv('REDDIT_CLIENT_SECRET'):
    raise Exception("Variáveis de ambiente do Reddit não configuradas!")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')

# Configura CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Configura Reddit API
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent="NEXUS Bot/1.0 by /u/nexus_bot"
)

try:
    # Verifica se CUDA está disponível
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível. GPU é necessária para executar este bot.")
    
    device = torch.device("cuda:0")
    logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    
    # Limpa cache da GPU
    torch.cuda.empty_cache()
    
    # Configura memória máxima
    max_memory = {0: "4GB"}
    
    # Carrega modelo Phi-2
    model_name = "microsoft/phi-2"
    
    # Configura tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Configura quantização em 4-bit para menor uso de memória
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Configura modelo com configurações otimizadas
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=True
    )
    
    # Cria pipeline otimizado
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    logger.info("✅ Modelo carregado com sucesso na GPU")
    
except Exception as e:
    logger.error(f"Erro ao carregar pipeline: {str(e)}")
    generator = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_reddit_posts(keyword, limit=10):
    try:
        # Busca posts
        subreddit = reddit.subreddit('all')
        posts = []
        
        for post in subreddit.search(keyword, sort='top', time_filter='month', limit=limit):
            if not post.stickied and not post.over_18 and post.score > 1000:  # Filtra apenas posts bem engajados
                posts.append({
                    'title': post.title,
                    'url': post.url if hasattr(post, 'url') else None,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'selftext': post.selftext[:500] if post.selftext else None
                })
        
        # Ordena por engajamento (score + comentários)
        posts.sort(key=lambda x: (x['score'] + x['num_comments']), reverse=True)
        
        logger.info(f"✅ Encontrados {len(posts)} posts sobre '{keyword}'")
        return posts[:3]  # Retorna apenas os 3 mais engajados
        
    except Exception as e:
        logger.error(f"❌ Erro ao buscar posts do Reddit: {str(e)}")
        raise

def generate_instagram_post(posts):
    try:
        if not generator:
            raise Exception("Modelo não foi carregado corretamente")

        # Cria prompt para o modelo gerar conteúdo para Instagram
        prompt = f"""Você é um especialista em criação de conteúdo para Instagram e tradução de inglês para português.

Aqui estão 3 posts populares do Reddit em inglês. Primeiro traduza cada um para português, depois crie uma publicação para o Instagram baseada neles:

Post 1:
Título original: {posts[0]['title']}
Conteúdo original: {posts[0]['selftext'][:200] if posts[0]['selftext'] else ''}

Post 2:
Título original: {posts[1]['title']}
Conteúdo original: {posts[1]['selftext'][:200] if posts[1]['selftext'] else ''}

Post 3:
Título original: {posts[2]['title']}
Conteúdo original: {posts[2]['selftext'][:200] if posts[2]['selftext'] else ''}

Primeiro, forneça a tradução de cada post no seguinte formato:

TRADUÇÕES:
Post 1:
[Tradução do título e conteúdo do post 1]

Post 2:
[Tradução do título e conteúdo do post 2]

Post 3:
[Tradução do título e conteúdo do post 3]

Em seguida, crie uma publicação para Instagram em português que:
1. Tenha uma introdução cativante
2. Resuma os principais pontos dos posts traduzidos de forma envolvente
3. Use emojis adequadamente
4. Inclua 3-5 hashtags relevantes em português
5. Termine com uma call-to-action em português
6. Mantenha o texto dentro do limite de caracteres do Instagram
7. Use quebras de linha estratégicas para melhor leitura

PUBLICAÇÃO PARA INSTAGRAM:
[Seu texto aqui]

.
.
.

#hashtag1 #hashtag2 #hashtag3"""

        # Gera o post usando o modelo
        formatted_prompt = f"Instruct: {prompt}\nOutput:"
        outputs = generator(
            formatted_prompt,
            max_new_tokens=1024,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = outputs[0]['generated_text']
        response = response.replace(formatted_prompt, '').strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Erro ao gerar post para Instagram: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        keyword = request.form.get('prompt')
        if keyword:
            try:
                # Busca posts do Reddit
                reddit_posts = get_reddit_posts(keyword)
                
                # Gera post para Instagram com traduções
                generated_content = generate_instagram_post(reddit_posts)
                
                return render_template('result.html', 
                                    generated_content=generated_content,
                                    reddit_posts=reddit_posts)
            except Exception as e:
                flash(f"Erro ao processar requisição: {str(e)}", "error")
                return render_template('index.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
