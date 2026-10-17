import os
import io
import asyncio
import requests
import discord
from discord.ext import commands
from threading import Thread
from flask import Flask

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------
# 1. Configurar Bot de Discord y Servidor Flask
# ---------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot de LinkedIn Activo 24/7", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

    # ---------------------------------------------------------
    # 2. Configurar Modelos de IA (Gemini + ChatGPT)
    # ---------------------------------------------------------
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    llm_gpt = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# ---------------------------------------------------------
# 3. Generador de Imagen Automático
# ---------------------------------------------------------
def generar_imagen_ejecutiva(prompt_idea: str) -> bytes:
    clean_prompt = requests.utils.quote(prompt_idea[:100])
    image_url = f"https://pollinations.ai/p/Minimalist%20modern%20executive%20illustration%20for%20LinkedIn%20about%20{clean_prompt}?width=1024&height=1024&seed=42"

    response = requests.get(image_url, timeout=30)
    if response.status_code == 200:
        return response.content
    raise Exception("No se pudo obtener la imagen.")

# ---------------------------------------------------------
# 4. Funciones para publicar en LinkedIn
# ---------------------------------------------------------
def registrar_y_subir_imagen_linkedin(image_bytes: bytes, token: str, author: str) -> str:
    register_url = "https://api.linkedin.com/v2/assets?action=registerUpload"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }

    register_payload = {
        "registerUploadRequest": {
        "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
        "owner": author,
        "serviceRelationships": [{"relationshipType": "OWNER", "identifier": "urn:li:userGeneratedContent"}]
    }
    }

    reg_res = requests.post(register_url, headers=headers, json=register_payload)
    if reg_res.status_code != 200:
        return None

    reg_data = reg_res.json()["value"]
    upload_url = reg_data["uploadMechanism"]["com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"]["uploadUrl"]
    asset_urn = reg_data["asset"]

    upload_headers = {"Authorization": f"Bearer {token}"}
    up_res = requests.put(upload_url, headers=upload_headers, data=image_bytes)

    return asset_urn if up_res.status_code in [200, 201] else None


def publicar_en_linkedin(texto_post: str, image_bytes_list: list = None) -> bool:
    url = "https://api.linkedin.com/v2/ugcPosts"
    token = os.getenv("LINKEDIN_ACCESS_TOKEN")
    author = os.getenv("LINKEDIN_PERSON_URN")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }

    media_list = []

    if image_bytes_list:
        for img_bytes in image_bytes_list:
            asset_urn = registrar_y_subir_imagen_linkedin(img_bytes, token, author)
            if asset_urn:
                media_list.append({
                    "status": "READY",
                    "description": {"text": "Imagen del post"},
                    "media": asset_urn,
                    "title": {"text": "Imagen"}
                })

        share_category = "IMAGE" if media_list else "NONE"

    payload = {
        "author": author,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
        "com.linkedin.ugc.ShareContent": {
        "shareCommentary": {"text": texto_post},
        "shareMediaCategory": share_category,
        "media": media_list if media_list else []
    }
    },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
    }

    res = requests.post(url, headers=headers, json=payload)
    return res.status_code == 201

# ---------------------------------------------------------
# 5. Flujo Multiagente CrewAI
# ---------------------------------------------------------
def ejecutar_orquestacion_agentes(idea_usuario: str) -> str:
    redactor = Agent(
        role='Redactor Ejecutivo de TI',
        goal='Crear borradores para LinkedIn basados en la experiencia del usuario.',
        backstory='Eres un líder de TI experimentado. Escribes con autoridad, de forma directa y enfocada en valor de negocio.',
        llm=llm_gemini,
        verbose=True
    )

    critico = Agent(
        role='Auditor Crítico Anti-IA',
        goal='Detectar y eliminar muletillas robóticas, frases hechas y clichés corporativos.',
        backstory='Aborreces los textos genéricos de IA. Exiges ganchos fuertes, oraciones cortas y tono 100% humano.',
        llm=llm_gpt,
        verbose=True
    )

    pulidor = Agent(
        role='Editor Final de Estilo',
        goal='Integrar las críticas y entregar la versión definitiva del post.',
        backstory='Especialista en pulir el mensaje conservando la autenticidad del autor.',
        llm=llm_gemini,
        verbose=True
    )

    t1 = Task(
        description=f'Escribe un borrador de post para LinkedIn sobre esta idea: {idea_usuario}',
        expected_output='Borrador completo estructurado.',
        agent=redactor
    )

    t2 = Task(
        description='Audita el borrador anterior. Señala qué frases suenan sintéticas o cliché y propone mejoras.',
        expected_output='Lista de correcciones críticas.',
        agent=critico
    )

    t3 = Task(
        description='Reescribe el post aplicando las correcciones del auditor. Devuelve SOLO el texto final listo para publicar.',
        expected_output='Post definitivo para LinkedIn.',
        agent=pulidor
    )

    crew = Crew(
        agents=[redactor, critico, pulidor],
        tasks=[t1, t2, t3],
        process=Process.sequential
    )

    return str(crew.kickoff())

# ---------------------------------------------------------
# 6. Interfaz de Bot en Discord con Botones
# ---------------------------------------------------------
class ComponentesAprobacion(discord.ui.View):
    def __init__(self, texto_post: str, image_bytes_list: list):
        super().__init__(timeout=None)
        self.texto_post = texto_post
        self.image_bytes_list = image_bytes_list

    @discord.ui.button(label="Aprobar y Publicar", style=discord.ButtonStyle.green, emoji="🚀")
    async def publicar(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        loop = asyncio.get_event_loop()
        exito = await loop.run_in_executor(None, publicar_en_linkedin, self.texto_post, self.image_bytes_list)
        if exito:
            await interaction.followup.send("✅ **¡Publicado con éxito en LinkedIn con la imagen!**")
        else:
            await interaction.followup.send("❌ **Error al publicar en LinkedIn.** Revisa tus credenciales.")

    @discord.ui.button(label="Descartar", style=discord.ButtonStyle.red, emoji="🗑️")
    async def descartar(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("❌ **Borrador descartado.**")

    # ---------------------------------------------------------
    # 7. Comando principal del Bot
    # ---------------------------------------------------------
    @bot.command(name="post")
    async def comando_post(ctx, *, idea: str = ""):
        await ctx.send("⚙️ *Orquestando agentes y procesando imagen...*")

        images_to_upload = []

        if ctx.message.attachments:
            for attachment in ctx.message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'webp']):
                    img_bytes = await attachment.read()
                images_to_upload.append(img_bytes)
                await ctx.send("📷 *Imagen(es) recibida(s) desde tu galería.*")

        else:
            await ctx.send("🎨 *No adjuntaste imagen. Generando ilustración profesional...*")
        try:
            loop = asyncio.get_event_loop()
            img_bytes = await loop.run_in_executor(None, generar_imagen_ejecutiva, idea)
            images_to_upload.append(img_bytes)
        except Exception as e:
            await ctx.send(f"⚠️ *No se pudo generar la imagen ({e}). Se procesará solo texto.*")

        # Ejecutar la orquestación en un ejecutor asíncrono para evitar bloqueos
        loop = asyncio.get_event_loop()
        post_final = await loop.run_in_executor(None, ejecutar_orquestacion_agentes, idea)

        vista = ComponentesAprobacion(texto_post=post_final, image_bytes_list=images_to_upload)

        if images_to_upload:
            file_preview = discord.File(fp=io.BytesIO(images_to_upload[0]), filename="preview.png")
            await ctx.send(f"📝 **Propuesta de Post Final:**\n\n{post_final}", file=file_preview, view=vista)
        else:
            await ctx.send(f"📝 **Propuesta de Post Final:**\n\n{post_final}", view=vista)

# ---------------------------------------------------------
# 8. Iniciar Servidores
# ---------------------------------------------------------
if __name__ == "__main__":
    Thread(target=run_flask, daemon=True).start()
    bot.run(os.getenv("DISCORD_BOT_TOKEN"))
