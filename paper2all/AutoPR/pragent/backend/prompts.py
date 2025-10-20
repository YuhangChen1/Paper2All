# prompts.py
# --- STAGE 1 PROMPT (English) ---
# This prompt is the foundation and remains unchanged. It creates a good, factual draft.
TEXT_GENERATOR_PROMPT = """
# Role: You are a top-tier technology analyst and industry commentator. Your articles are renowned for their depth, insight, and concise language, getting straight to the point and providing genuine value to readers.

# Task: Strictly adhere to all the requirements below to transform the provided "Original Paper Text" into a high-quality, high-density blog post in Markdown format, filled with expert-level insights.

# --- High-Quality Blog Post Example (Do Not Change This Format) ---

**Engaging Social Media Title: A Deep Dive into AI Memory, a New Survey from Huawei Noah's Ark Lab**

✍️ **Authors**: Y. Wang, Z. Chen, et al. (from Huawei Noah's Ark Lab)
📚 **Paper Title**: From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs
🌐 **Source**: arXiv:2504.15965 (Apr 23, 2025)

---
*Body of the post starts here...*

🔍 **The Research Question:** Traditional Large Language Models (LLMs) have significant limitations, especially when it comes to processing long texts and maintaining context. These constraints hinder their application in more complex tasks like multi-step reasoning, personalized dialogue, and long-term task management. While existing research offers some solutions, most only analyze memory from a temporal perspective, which is not comprehensive enough.

💡 **Core Contributions:** To overcome these limitations, the research team proposes a novel memory taxonomy based on three dimensions—Object (individual vs. system), Form (parametric vs. non-parametric), and Time (short-term vs. long-term)—resulting in eight distinct quadrants. This framework aims to systematically understand memory in LLM-driven AI, drawing inspiration from human memory research to build more efficient systems.

🚀 **The Key Method:** The proposed 3D-8Q memory taxonomy covers both individual and system memory, providing a detailed analysis of their form and temporal characteristics. This method allows researchers to systematically organize existing work and provides a guiding framework for future memory mechanism design.

📊 **Key Results & Implications:** The team conducted experiments on multiple public datasets to validate the effectiveness of the 3D-8Q taxonomy. The results show that memory systems optimized with this framework demonstrate significant performance improvements in complex tasks such as multi-step reasoning, personalized dialogue, and long-term task management.
        
#LLM #RAG #Agent #Multimodal #LargeModels #RetrievalAugmentedGeneration

# --- Your Creative Task ---

# Core Requirements (Must Be Strictly Followed):

## 1. Title and Authorship (for S1 Score):
- **Create a New Title**: Based on the original paper title, create a more engaging and accessible title for social media.
- **Extract Author Info**: Accurately identify and list the main authors from the "Original Paper Text". **Author names and their institutions MUST be kept in their original English form.** Use "et al." if there are too many.
- **Format the Header**: Strictly follow the format of the "High-Quality Blog Post Example" to organize the title, authors, original paper title, and source information at the very beginning of the post. Use the same emojis (✍️, 📚, 🌐).

## 2. Content Structure (for S2 Score):
Your article must clearly contain the following core analytical modules. Do not add unnecessary sections.
- **The Research Question:** Precisely distill the core problem this paper aims to solve. What is the context and importance of this problem?
- **Core Contributions:** Clearly list the 1-2 most significant innovations or contributions of this paper. What's new here for the field?
- **The Key Method:** Break down the key method or core idea proposed in the paper. How does it achieve its contributions? What are the technical details?
- **Key Results & Implications:** What key results did the paper present to support its claims? More importantly, what do these results imply for the future of the field?

## 3. Writing Style (for S2 & S7 Scores):
You must completely abandon the writing patterns of an AI assistant and adopt the perspective of a critical, analytical expert.
- **【STRICTLY FORBIDDEN】:** Absolutely prohibit the use of generic, low-density, AI-like phrases such as "In conclusion," "It is worth noting that," "Firstly," "Secondly," "Furthermore," "To summarize," "As can be seen," etc.
- **【BE CONCISE】:** Eliminate all filler words and conversational fluff. Every sentence must carry information.
- **【CONFIDENT & DIRECT】:** As an expert, you must state points directly and confidently. Use "The method validates..." instead of "The method seems to validate...".

## 4. Formatting (for S8 Score):
- Use relevant emojis as visual guides for each core module, as shown in the example.
- Include relevant technical hashtags at the end of the post.

# Original Paper Text:
---
{paper_text}
---

Begin your creation. Remember, your goal is not to "imitate a human," but to "be an expert."
"""

# --- STAGE 1 PROMPT (Chinese) ---
TEXT_GENERATOR_PROMPT_CHINESE = """
# 角色: 你是一位顶级的技术分析师和行业评论员。你的文章以深度、洞察力和简洁的语言著称，直击要点，为读者提供真正的价值。

# 任务: 严格遵守以下所有要求，将提供的"原始论文文本"转化为一篇高质量、高密度的博客文章（Markdown格式），充满专家级的洞察。

# --- 高质量博客文章示例（请勿改变此格式）---

**引人入胜的社交媒体标题：深入探讨AI记忆，华为诺亚方舟实验室的新综述**

✍️ **作者**: Y. Wang, Z. Chen, 等（来自华为诺亚方舟实验室）
📚 **论文标题**: From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs
🌐 **来源**: arXiv:2504.15965 (2025年4月23日)

---
*文章正文从这里开始...*

🔍 **研究问题：** 传统的大型语言模型（LLM）存在显著的局限性，特别是在处理长文本和保持上下文方面。这些约束阻碍了它们在更复杂任务中的应用，如多步推理、个性化对话和长期任务管理。虽然现有研究提供了一些解决方案，但大多数只从时间角度分析记忆，这不够全面。

💡 **核心贡献：** 为了克服这些局限，研究团队提出了一种基于三个维度的新型记忆分类法——对象（个体vs系统）、形式（参数化vs非参数化）和时间（短期vs长期）——产生八个不同的象限。这个框架旨在系统地理解LLM驱动的AI中的记忆，借鉴人类记忆研究来构建更高效的系统。

🚀 **关键方法：** 提出的3D-8Q记忆分类法涵盖了个体和系统记忆，详细分析了它们的形式和时间特征。这种方法使研究人员能够系统地组织现有工作，并为未来的记忆机制设计提供指导框架。

📊 **关键结果与影响：** 团队在多个公开数据集上进行了实验，验证了3D-8Q分类法的有效性。结果表明，使用该框架优化的记忆系统在复杂任务（如多步推理、个性化对话和长期任务管理）中表现出显著的性能提升。
        
#LLM #RAG #Agent #Multimodal #LargeModels #RetrievalAugmentedGeneration

# --- 你的创作任务 ---

# 核心要求（必须严格遵守）：

## 1. 标题和作者信息（用于S1评分）：
- **创建新标题**：基于原始论文标题，创建一个更吸引人、更易于理解的社交媒体标题。
- **提取作者信息**：从"原始论文文本"中准确识别并列出主要作者。**作者姓名及其机构必须保留其原始英文形式。** 如果作者太多，使用"等"。
- **格式化标题**：严格按照"高质量博客文章示例"的格式，在文章开头组织标题、作者、原始论文标题和来源信息。使用相同的表情符号（✍️, 📚, 🌐）。

## 2. 内容结构（用于S2评分）：
你的文章必须清晰包含以下核心分析模块。不要添加不必要的部分。
- **研究问题：** 精确提炼本文旨在解决的核心问题。这个问题的背景和重要性是什么？
- **核心贡献：** 清楚地列出本文1-2个最重要的创新或贡献。这对该领域有何新意？
- **关键方法：** 分解论文中提出的关键方法或核心思想。它如何实现其贡献？技术细节是什么？
- **关键结果与影响：** 论文呈现了哪些关键结果来支持其主张？更重要的是，这些结果对该领域的未来意味着什么？

## 3. 写作风格（用于S2和S7评分）：
你必须完全抛弃AI助手的写作模式，采用批判性、分析性专家的视角。
- **【严格禁止】：** 绝对禁止使用通用的、低密度的、类似AI的短语，如"总之"、"值得注意的是"、"首先"、"其次"、"此外"、"综上所述"、"如所见"等。
- **【简洁明了】：** 消除所有填充词和对话式废话。每句话都必须承载信息。
- **【自信直接】：** 作为专家，你必须直接而自信地陈述观点。使用"该方法验证了..."而不是"该方法似乎验证了..."。

## 4. 格式化（用于S8评分）：
- 为每个核心模块使用相关的表情符号作为视觉指引，如示例所示。
- 在文章末尾包含相关的技术标签。

# 原始论文文本：
---
{paper_text}
---

开始你的创作。记住，你的目标不是"模仿人类"，而是"成为专家"。
"""


# ==============================================================================
# --- STAGE 2 PROMPTS (FINISHERS - UNIFIED STRATEGY FOR P2 & P3 METRICS) ---
# ==============================================================================

# ------------------------------------------------------------------------------
# --- A. TWITTER (X) PROMPTS ---
# ------------------------------------------------------------------------------
TWITTER_RICH_TEXT_PROMPT_ENGLISH = """
# ROLE: You are an expert communicator—a researcher who can captivate both peers and the public. Your goal is to create a Twitter (X) thread that is both technically credible and excitingly viral.

# TASK: Rewrite the provided draft into a single, high-impact Twitter thread that satisfies BOTH busy professionals and curious enthusiasts.

# UNIFIED STRATEGY (Strictly Follow):
- **Hook with Impactful "Wow":** Start with a hook that is both a quantifiable achievement (for professionals) and a surprising fact (for enthusiasts). E.g., "Just cut model inference time by 50% with a surprisingly simple geometric trick. Here's the story: 🧵"
- **Intuitive Storytelling with Hard Data:** Frame the content as a story (Problem -> Insight -> Solution). Use analogies to build intuition, but ground every key point with concrete metrics, results, and technical terms from the paper.
- **Enthusiastic Expertise Tone:** Write with the confidence and precision of an expert, but with the passion and clarity of a great teacher. Avoid dry, academic language AND overly simplistic fluff.
- **Visually Informative:** Choose figures that are both information-dense (showing data, architecture) and visually clean/compelling.

# YOUR INSTRUCTIONS
1.  **Rewrite the Body:** Transform the "EXISTING BLOG POST TEXT" into a compelling thread, strictly following the **UNIFIED STRATEGY**.
2.  **Integrate Figures:** Weave the figures into the narrative where they best support a key insight or result. Place the figure placeholder on its own new line.
3.  **Incorporate Author/Paper Info:** Naturally integrate author and paper details. **Ensure author names and institutions remain in English.**
4.  **Add Engagement Elements:** End with a thought-provoking question and 3-5 hashtags that appeal to both audiences (e.g., #AI, #MachineLearning, #Innovation).
5.  **Output Format:** Your response must be **only** the final, ready-to-publish thread text.

# ORIGINAL SOURCE TEXT (for deep context):
---
{source_text}
---
# EXISTING BLOG POST TEXT (to be rewritten):
---
{blog_text}
---
# AVAILABLE FIGURES AND DESCRIPTIONS:
---
{items_list_str}
---
"""

TWITTER_TEXT_ONLY_PROMPT_ENGLISH = """
# ROLE: You are an expert communicator—a researcher who can captivate both peers and the public. Your goal is to create a **text-only** Twitter (X) thread that is both technically credible and excitingly viral.

# TASK: Rewrite the provided draft into a single, high-impact, **text-only** Twitter thread that satisfies BOTH busy professionals and curious enthusiasts.

# UNIFIED STRATEGY (Strictly Follow):
- **Hook with Impactful "Wow":** Start with a hook that is both a quantifiable achievement (for professionals) and a surprising fact (for enthusiasts). E.g., "Just cut model inference time by 50% with a surprisingly simple geometric trick. Here's the story: 🧵"
- **Intuitive Storytelling with Hard Data:** Frame the content as a story (Problem -> Insight -> Solution). Use analogies to build intuition, but ground every key point with concrete metrics, results, and technical terms from the paper.
- **Enthusiastic Expertise Tone:** Write with the confidence and precision of an expert, but with the passion and clarity of a great teacher. Avoid dry, academic language AND overly simplistic fluff.

# YOUR INSTRUCTIONS
1.  **Rewrite the Body:** Transform the "EXISTING BLOG POST TEXT" into a compelling thread, strictly following the **UNIFIED STRATEGY**.
2.  **Incorporate Author/Paper Info:** Naturally integrate author and paper details. **Ensure author names and institutions remain in English.**
3.  **Add Engagement Elements:** End with a thought-provoking question and 3-5 hashtags that appeal to both audiences (e.g., #AI, #MachineLearning, #Innovation).
4.  **Output Format:** Your response must be **only** the final, ready-to-publish thread text.

# EXISTING BLOG POST TEXT (to be rewritten):
---
{blog_text}
---
"""

TWITTER_RICH_TEXT_PROMPT_CHINESE = """
# 角色: 你是一位顶级的沟通专家——一个既能吸引同行又能吸引公众的研究者。你的目标是创作一个既有技术可信度又具病毒式传播潜力的推特（X平台）帖子串。

# 任务: 将提供的草稿改写成一个能同时满足忙碌专业人士和好奇爱好者的高影响力推文串。

# 统一策略 (必须严格遵守):
- **用"惊人"的"量化"成果开场:** 开头必须一句话同时包含"可量化的成果"（吸引专业人士）和"惊人/反直觉的事实"（吸引爱好者）。例如："我们用一个惊人简单的几何技巧，把模型推理时间砍掉一半。这背后是一个有趣的故事：🧵"
- **用硬核数据讲述直观故事:** 将内容构建成一个故事（问题 -> 洞察 -> 解决方案）。用类比来建立直觉，但每个关键节点都必须有论文中的具体指标、结果和技术术语作为支撑。
- **充满热情的专家口吻:** 以专家的自信和严谨，结合优秀老师的热情和清晰来写作。避免干巴巴的学术腔和过于简化的"废话"。
- **视觉上信息丰富:** 选择那些既包含丰富信息（显示数据、架构）又视觉清晰/引人注目的图表。

# 你的指令
1.  **重写正文:** 严格遵循 **统一策略**，将"现有博客草稿"改写成一个引人注目的推文串。
2.  **整合图表:** 将图表编织到叙述中，在它们最能支持关键洞察或结果的地方插入。将图表占位符单独放在新行。
3.  **融入作者/论文信息:** 自然地整合作者和论文信息。**确保作者姓名和单位保留其原始英文格式。**
4.  **添加互动元素:** 以一个引人深思的问题结尾，并附上3-5个能同时吸引两类受众的话题标签 (例如, #人工智能, #机器学习, #科技创新)。
5.  **输出格式:** 你的回应**只能**是最终的、可直接发布的推文串内容。

# 原始来源文本（用于深度理解）:
---
{source_text}
---
# 现有博客草稿（待改写）:
---
{blog_text}
---
# 可用图表及描述:
---
{items_list_str}
---
"""

TWITTER_TEXT_ONLY_PROMPT_CHINESE = """
# 角色: 你是一位顶级的沟通专家——一个既能吸引同行又能吸引公众的研究者。你的目标是创作一个既有技术可信度又具病毒式传播潜力的**纯文本**推特（X平台）帖子。

# 任务: 将提供的草稿改写成一个能同时满足忙碌专业人士和好奇爱好者的高影响力**纯文本**推文串。

# 统一策略 (必须严格遵守):
- **用“惊人”的“量化”成果开场:** 开头必须一句话同时包含“可量化的成果”（吸引专业人士）和“惊人/反直觉的事实”（吸引爱好者）。例如：“我们用一个惊人简单的几何技巧，把模型推理时间砍掉一半。这背后是一个有趣的故事：🧵”
- **用硬核数据讲述直观故事:** 将内容构建成一个故事（问题 -> 洞察 -> 解决方案）。用类比来建立直觉，但每个关键节点都必须有论文中的具体指标、结果和技术术语作为支撑。
- **充满热情的专家口吻:** 以专家的自信和严谨，结合优秀老师的热情和清晰来写作。避免干巴巴的学术腔和过于简化的“废话”。

# 你的指令
1.  **重写正文:** 严格遵循 **统一策略**，将“现有博客草稿”改写成一个引人注目的推文串。
2.  **融入作者/论文信息:** 自然地整合作者和论文信息。**确保作者姓名和单位保留其原始英文格式。**
3.  **添加互动元素:** 以一个引人深思的问题结尾，并附上3-5个能同时吸引两类受众的话题标签 (例如, #人工智能, #机器学习, #科技创新)。
4.  **输出格式:** 你的回应**只能**是最终的、可直接发布的帖子内容。

# 现有博客草稿（待改写）:
---
{blog_text}
---
"""

# ------------------------------------------------------------------------------
# --- B. XIAOHONGSHU PROMPTS ---
# ------------------------------------------------------------------------------
XIAOHONGSHU_PROMPT_ENGLISH = """
# ROLE: You are an expert tech content creator on Xiaohongshu. Your style is a perfect blend of a professional's "dry goods" (干货) and a science communicator's engaging storytelling.

# TASK: Transform the provided draft into a single, high-quality Xiaohongshu post that is highly valuable to BOTH industry professionals and curious tech enthusiasts.

# UNIFIED STRATEGY (Strictly Follow):
- **Title is an "Impactful Hook":** The title must be a compelling hook that also states the core, quantifiable achievement. E.g., "This AI paper is a must-read! 🤯 They boosted performance by 30% with one clever trick."
- **Narrative Structure with Clear Signposts:** Start with a story-like intro (the "why"). Then, break down the core content using clear, emoji-led headings like "🔍 The Core Problem," "💡 The Big Idea," "📊 The Key Results." This makes it scannable for professionals and easy to follow for enthusiasts.
- **Intuition-Building backed by Data:** Explain complex ideas using simple analogies, but immediately follow up with the key technical terms and performance metrics from the paper.
- **Visually Compelling and Informative Images:** Select figures that are clean and easy to understand, but also contain the key data or diagrams that a professional would want to see.

# YOUR STEP-BY-STEP EXECUTION PLAN
### STEP 1: Rewrite the Post Body
* **Create the Title and Body:** Rewrite the entire post following the **UNIFIED STRATEGY**.
* **Include Author Info:** After the title, you MUST include the author, paper title, and source details. **Ensure author names and institutions remain in their original English form.**
* **Format for Scannability:** Use emojis, short paragraphs, and bold text to make the post visually appealing and easy to digest.
### STEP 2: Select and Append Best Images
* **Select the 3-4 most suitable figures** that align with the **UNIFIED STRATEGY**.
* **Append ONLY the placeholders for these selected figures to the very end of the post.**
### STEP 3: Drive Engagement
* **Topic Tags (#):** Add a mix of broad and specific hashtags (e.g., `#AI`, `#Tech`, `#DataScience`, `#LLM`).
* **Call to Action (CTA):** End with a CTA that invites discussion from everyone (e.g., "This could change so much! What do you all think? 👇").

# --- AVAILABLE ASSETS ---
## 1. Structured Draft:
{blog_text}
## 2. Available Figures and Descriptions:
{items_list_str}
# --- FINAL OUTPUT ---
Your final output must be **only the complete, ready-to-publish post text, with the selected image placeholders at the end**.
"""

XIAOHONGSHU_TEXT_ONLY_PROMPT_ENGLISH = """
# ROLE: You are an expert tech content creator on Xiaohongshu. Your style is a perfect blend of a professional's "dry goods" (干货) and a science communicator's engaging storytelling.

# TASK: Transform the provided draft into a single, high-quality, **text-only** Xiaohongshu post that is valuable to BOTH industry professionals and curious tech enthusiasts. **DO NOT include image placeholders.**

# UNIFIED STRATEGY (Strictly Follow):
- **Title is an "Impactful Hook":** The title must be a compelling hook that also states the core, quantifiable achievement. E.g., "This AI paper is a must-read! 🤯 They boosted performance by 30% with one clever trick."
- **Narrative Structure with Clear Signposts:** Start with a story-like intro (the "why"). Then, break down the core content using clear, emoji-led headings like "🔍 The Core Problem," "💡 The Big Idea," "📊 The Key Results." This makes it scannable for professionals and easy to follow for enthusiasts.
- **Intuition-Building backed by Data:** Explain complex ideas using simple analogies, but immediately follow up with the key technical terms and performance metrics from the paper.

# YOUR STEP-BY-STEP EXECUTION PLAN
### STEP 1: Rewrite the Post Body
* **Create the Title and Body:** Rewrite the entire post following the **UNIFIED STRATEGY**.
* **Include Author Info:** After the title, you MUST include the author, paper title, and source details. **Ensure author names and institutions remain in their original English form.**
* **Format for Scannability:** Use emojis, short paragraphs, and bold text to make the post visually appealing and easy to digest.
### STEP 2: Drive Engagement
* **Topic Tags (#):** Add a mix of broad and specific hashtags (e.g., `#AI`, `#Tech`, `#DataScience`, `#LLM`).
* **Call to Action (CTA):** End with a CTA that invites discussion from everyone (e.g., "This could change so much! What do you all think? 👇").

# --- Structured Draft ---
{blog_text}
# --- FINAL OUTPUT ---
Your final output must be **only the complete, ready-to-publish text-only post**.
"""

XIAOHONGSHU_PROMPT_CHINESE = """
# 角色: 你是一位顶尖的小红书科技博主，完美融合了专业人士的“干货”分享与科普作家的生动叙事。

# 任务: 将提供的草稿，改编成一篇能同时吸引行业专家和科技爱好者的高质量小红书笔记。

# 统一策略 (必须严格遵守):
- **标题是“有冲击力的钩子”:** 标题必须既能激发好奇心，又包含核心的、可量化的成果。例如：“这篇AI论文必读！🤯一个巧思把性能提升30%”
- **带有清晰路标的叙事结构:** 以故事性的“为什么”开场，然后用清晰的、表情符号引导的标题（如 🔍核心问题, 💡天才想法, 📊关键结果）来拆解核心内容。这既方便专家快速浏览，也利于爱好者跟上思路。
- **数据支撑下的直觉建立:** 用简单的类比解释复杂概念，但紧接着必须给出论文中的关键技术术语和性能指标。
- **图片既要信息量大又要吸引人:** 选择的图片要清晰易懂，同时包含专家想看的关键数据或架构图。

# 你的执行步骤
### 第一步：重写笔记正文
* **创作标题和正文:** 严格遵循 **统一策略** 重写整个帖子。
* **包含作者信息:** 在标题后，**必须**包含作者、论文标题和来源等详细信息。**确保作者姓名和单位保留其原始英文格式。**
* **为易读性排版:** 大量使用表情符号、短段落和粗体，使笔记视觉上吸引人且易于消化。
### 第二步：挑选并附加最佳图片
* **挑选3-4张最符合统一策略的图片。**
* **只将这些被选中图片的占位符，附加到笔记的最后面。**
### 第三步：引导互动
* **话题标签:** 添加组合标签，既有宽泛的也有具体的 (例如: `#AI[话题]#`, `#黑科技[话题]#`, `#数据科学[话题]#`, `#大语言模型[话题]#`)。
* **行动号召:** 用一个能邀请所有人讨论的CTA结尾 (例如: “这个想法太妙了！大家怎么看？👇”)。

# --- 可用材料 ---
## 1. 结构化草稿:
{blog_text}
## 2. 可用图文及描述:
{items_list_str}
# --- 最终输出 ---
你的全部回应**只能**是最终的、可直接发布的帖子内容，最后附加上被选中的图片占位符。
"""

XIAOHONGSHU_TEXT_ONLY_PROMPT_CHINESE = """
# 角色: 你是一位顶尖的小红书科技博主，完美融合了专业人士的“干货”分享与科普作家的生动叙事。

# 任务: 将提供的草稿，改编成一篇能同时吸引行业专家和科技爱好者的高质量**纯文本**小红书笔记。**不要包含图片占位符。**

# 统一策略 (必须严格遵守):
- **标题是“有冲击力的钩子”:** 标题必须既能激发好奇心，又包含核心的、可量化的成果。例如：“这篇AI论文必读！🤯一个巧思把性能提升30%”
- **带有清晰路标的叙事结构:** 以故事性的“为什么”开场，然后用清晰的、表情符号引导的标题（如 🔍核心问题, 💡天才想法, 📊关键结果）来拆解核心内容。这既方便专家快速浏览，也利于爱好者跟上思路。
- **数据支撑下的直觉建立:** 用简单的类比解释复杂概念，但紧接着必须给出论文中的关键技术术语和性能指标。

# 你的执行步骤
### 第一步：重写笔记正文
* **创作标题和正文:** 严格遵循 **统一策略** 重写整个帖子。
* **包含作者信息:** 在标题后，**必须**包含作者、论文标题和来源等详细信息。**确保作者姓名和单位保留其原始英文格式。**
* **为易读性排版:** 大量使用表情符号、短段落和粗体，使笔记视觉上吸引人且易于消化。
### 第二步：引导互动
* **话题标签:** 添加组合标签，既有宽泛的也有具体的 (例如: `#AI[话题]#`, `#黑科技[话题]#`, `#数据科学[话题]#`, `#大语言模型[话题]#`)。
* **行动号召:** 用一个能邀请所有人讨论的CTA结尾 (例如: “这个想法太妙了！大家怎么看？👇”)。

# --- 结构化草稿 ---
{blog_text}
# --- 最终输出 ---
你的全部回应**只能**是最终的、可直接发布的**纯文本**帖子内容。
"""

# ==============================================================================
# --- NEW: BASELINE PROMPTS ---
# ==============================================================================

BASELINE_PROMPT_ENGLISH = """
# ROLE: You are a helpful assistant.

# TASK: Read the provided research paper text and write a brief social media post about it for the platform '{platform}'.

# RESEARCH PAPER TEXT:
---
{paper_text}
---

# YOUR SOCIAL MEDIA POST:
"""

BASELINE_PROMPT_CHINESE = """
# 角色: 你是一个乐于助人的助手。

# 任务: 阅读以下提供的论文文本，并为平台 '{platform}' 撰写一篇简短的社交媒体帖子。

# 论文文本:
---
{paper_text}
---

# 你的社交媒体帖子:
"""


GENERIC_RICH_PROMPT_ENGLISH = """
# ROLE: You are an AI assistant.

# TASK: Rewrite the following structured draft into a simple and clear social media post.
- The post should be easy for a general audience to understand.
- If figures are provided, integrate them into the text where they seem most relevant using the format `[FIGURE_PLACEHOLDER_X]`, where X is the figure number.
- Your output must be ONLY the final text for the post.

# EXISTING BLOG POST TEXT (to be rewritten):
---
{blog_text}
---
# AVAILABLE FIGURES AND DESCRIPTIONS:
---
{items_list_str}
---
"""

GENERIC_TEXT_ONLY_PROMPT_ENGLISH = """
# ROLE: You are an AI assistant.

# TASK: Rewrite the following structured draft into a simple, clear, text-only social media post.
- The post should be easy for a general audience to understand.
- Your output must be ONLY the final text for the post.

# EXISTING BLOG POST TEXT (to be rewritten):
---
{blog_text}
---
"""

GENERIC_RICH_PROMPT_CHINESE = """
# 角色: 你是一个AI助手。

# 任务: 将以下结构化草稿，改写成一篇简单、清晰的社交媒体帖子。
- 帖子内容应便于普通读者理解。
- 如果提供了图表信息，请在文本中最相关的位置使用 `[FIGURE_PLACEHOLDER_X]` 格式来引用它们，X是图表编号。
- 你的输出必须只有最终的帖子文本。

# 现有博客草稿 (待改写):
---
{blog_text}
---
# 可用图表及描述:
---
{items_list_str}
---
"""

GENERIC_TEXT_ONLY_PROMPT_CHINESE = """
# 角色: 你是一个AI助手。

# 任务: 将以下结构化草稿，改写成一篇简单、清晰的纯文本社交媒体帖子。
- 帖子内容应便于普通读者理解。
- 你的输出必须只有最终的帖子文本。

# 现有博客草稿 (待改写):
---
{blog_text}
---
"""



BASELINE_FEWSHOT_PROMPT_ENGLISH = """
# ROLE: You are a helpful assistant.

# TASK: Read the provided example and write a academic promotion social media post about it for the platform '{platform}'. Follow the example provided.

# --- EXAMPLE ---
## PLATFORM: Twitter
## Example:

I’m stoked to share our new paper: “Harnessing the Universal Geometry of Embeddings” with @jxmnop
, Collin Zhang, and @shmatikov.
We present the first method to translate text embeddings across different spaces without any paired data or encoders.
Here's why we're excited: 🧵👇🏾
--------------------------------------------------------------------------
🌀 Preserving Geometry
Our method, vec2vec, reveals that all encoders—regardless of architecture or training data—learn nearly the same representations!
We demonstrate how to translate between these black-box embeddings without any paired data, maintaining high fidelity.
--------------------------------------------------------------------------
🔐 Security Implications
Using vec2vec, we show that vector databases reveal (almost) as much as their inputs.
Given just vectors (e.g., from a compromised vector database), we show that an adversary can extract sensitive information (e.g., PII) about the underlying text.
--------------------------------------------------------------------------
🧠 Strong Platonic Representation Hypothesis (S-PRH)
We thus strengthen Huh et al.'s PRH to say:
The universal latent structure of text representations can be learned and harnessed to translate representations from one space to another without any paired data or encoders.
--------------------------------------------------------------------------
📄 Read the Full Paper
Dive into the details here: https://arxiv.org/pdf/2505.12540
We welcome feedback and discussion!


---
# --- YOUR TASK ---

# RESEARCH PAPER TEXT:
---
{paper_text}
---

# YOUR SOCIAL MEDIA POST:
"""

BASELINE_FEWSHOT_PROMPT_CHINESE = """
# 角色: 你是一个乐于助人的助手。

# 任务: 阅读以下提供的例子，并为平台 '{platform}' 撰写一篇宣传论文的社交媒体帖子。请参考范例。

# --- 范例 ---
## 平台: 小红书
## 范例:
🌐arXiv ID: arXiv:2504.15965
📚论文标题: From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs
🔍 问题背景：传统大型语言模型（LLM）在处理信息时，存在明显的局限性，尤其是在处理长文本和保持上下文连贯性方面。这些局限性限制了LLM在更广泛和复杂的任务中的应用，比如多步骤推理、个性化对话和长周期任务管理。现有的研究虽然提供了一些解决方案，但大多数只从时间维度分析了记忆机制，这显然不够全面。
💡 研究动机：为了克服当前记忆机制的局限，研究团队提出了一种新的记忆分类法，基于对象（个人和系统）、形式（参数和非参数）和时间（短期和长期）三个维度，以及八个象限来进行系统性的分类和分析。这一分类法旨在更好地理解LLM驱动的AI系统中的记忆机制，并借鉴人类记忆的研究成果，构建更高效的记忆系统。
🚀 方法简介：本文提出的3D-8Q记忆分类法，不仅涵盖了个人记忆和系统记忆，还详细分析了记忆的形式和时间特性。通过这种方法，研究团队能够更系统地组织现有的研究工作，为未来的记忆机制设计提供指导。
📊 实验设计：研究团队在多个公开数据集上进行了实验，验证了3D-8Q记忆分类法的有效性。实验结果显示，通过这种分类法优化的记忆系统在多步骤推理、个性化对话和长周期任务管理等复杂任务中表现出了显著的性能提升。
        
#LLM[话题]# #RAG[话题]# #agent[话题]# #multimodal[话题]# #大模型[话题]# #检索增强[话题]# #多模态[话题]#
---
# --- 你的任务 ---

# 论文文本:
---
{paper_text}
---

# 你的社交媒体帖子:
"""