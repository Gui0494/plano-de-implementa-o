# plano-de-implementa-o 

# Plano Enterprise Completo: Construção de IA de Alto Nível

Este documento apresenta um plano enterprise extremamente detalhado para construção de sistemas de IA de codificação autônoma (como Devin, Cursor, Claude Code), cobrindo todas as camadas desde fundamentos de ciência da computação até troubleshooting completo.

---

## Sumário Executivo

Construir um assistente de codificação de IA de nível enterprise requer domínio de **16 pilares técnicos** interconectados. Os sistemas líderes do mercado (Cursor, Devin, Claude Code) demonstram que o sucesso depende de: arquitetura de agentes bem projetada, execução segura em sandbox, RAG otimizado para código, e observabilidade profunda. Este plano fornece arquiteturas de referência, checklists de implementação, padrões de código e estratégias de troubleshooting para cada camada.

**Métricas de sucesso dos líderes:**
- **Cursor**: 250 tokens/segundo, 8 agentes paralelos, latência sub-100ms
- **Devin**: 67% taxa de merge de PRs, 4x mais rápido que ano anterior
- **SWE-agent**: 65% no SWE-bench Verified em apenas 100 linhas de Python

---

## PARTE I: FUNDAMENTOS TÉCNICOS

### 1. Ciência da Computação para Sistemas de IA

#### 1.1 Estruturas de Dados Críticas

| Estrutura | Aplicação em IA | Complexidade | Recomendação |
|-----------|-----------------|--------------|--------------|
| **Hash Tables** | Cache semântico de embeddings e respostas LLM | O(1) lookup | Redis com similaridade vetorial; threshold 0.85-0.95 |
| **Árvores/Grafos** | Planejamento de agentes (Tree of Thoughts, DAGs) | O(log n) busca | HNSW para vizinhos mais próximos; ToT para raciocínio |
| **Priority Queues** | Scheduling de tarefas, continuous batching | O(log n) insert | vLLM para gerenciamento de requests |
| **Tries** | Tokenização eficiente (BPE) | O(m) lookup | Tokenizers HuggingFace |

**Padrão de Cache Semântico:**
```python
# Redis Semantic Cache - 50%+ redução de custos
from redisvl.extensions.cache.llm import SemanticCache
llmcache = SemanticCache(
    name="llmcache",
    distance_threshold=0.1,  # Menor = mais restritivo
    vectorizer=HFTextVectorizer("redis/langcache-embed-v1")
)
```

#### 1.2 Algoritmos para Agentes

**Tree of Thoughts vs Chain of Thought:**
- CoT: 4% accuracy no Game of 24
- ToT com BFS/DFS: **74%+ accuracy** (100x mais tokens)
- Graph of Thoughts: 33% menos erros que ToT

**HNSW para Busca Vetorial:**
```python
# Configuração HNSW otimizada
import hnswlib
index = hnswlib.Index(space='l2', dim=768)
index.init_index(
    max_elements=1_000_000,
    ef_construction=100,  # Maior = melhor recall, build mais lento
    M=16                  # Links por nó; maior = mais memória, melhor accuracy
)
index.set_ef(50)  # Parâmetro de busca
```

#### 1.3 Concorrência para LLMs

**Padrão Async para Chamadas LLM (10x improvement):**
```python
import asyncio
from openai import AsyncOpenAI

async def process_batch(prompts: list[str], batch_size: int = 10):
    client = AsyncOpenAI()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tasks = [client.chat.completions.create(...) for p in batch]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    return responses
```

#### 1.4 Arquitetura de Hardware para Inferência

| Acelerador | Throughput | Latência | Melhor Para |
|------------|-----------|----------|-------------|
| **CPU** | Baixo | Variável | Modelos pequenos, tráfego baixo |
| **GPU (NVIDIA)** | Muito Alto | Baixa | Treinamento + Inferência |
| **TPU** | Muito Alto | Baixa | Escala massiva (cloud) |

**Quantização - Trade-offs:**

| Formato | Redução Memória | Impacto Accuracy | Hardware |
|---------|-----------------|------------------|----------|
| BF16 | Baseline | Baseline | Todos modernos |
| FP8 | 2x | ~0.04% queda | H100, Ada Lovelace |
| INT8 | 2x | Mínimo | Ampere+ |
| INT4 | 4x | 1-2% queda | Requer calibração |

**Recomendação**: INT8 como default seguro; FP8 em H100 para melhor equilíbrio.

---

### 2. Deep Learning e Transformers

#### 2.1 Matemática da Atenção

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

- Complexidade: O(n²) para sequência de comprimento n
- **Flash Attention**: Reduz memória de O(n²) para O(n)
- Multi-Head: Permite atenção a diferentes aspectos simultaneamente

#### 2.2 Positional Encoding

| Método | Vantagens | Uso |
|--------|-----------|-----|
| **Sinusoidal** | Simples, sem parâmetros | Transformer original |
| **RoPE** | Posições relativas, sem parâmetros | LLaMA, GPT-NeoX |
| **ALiBi** | Melhor extrapolação | MPT, BLOOM |

**Recomendação**: RoPE para uso geral; ALiBi se extrapolação crítica.

#### 2.3 Scaling Laws (Chinchilla)

**Regra de ouro: tokens ≈ 20 × parâmetros**

| Tamanho Modelo | Tokens Ótimos | Dados |
|----------------|---------------|-------|
| 7B | 140B | ~500GB texto |
| 13B | 260B | ~1TB texto |
| 70B | 1.4T | ~5TB texto |

#### 2.4 Fine-Tuning Eficiente

| Método | Parâmetros Treináveis | Memória | Performance |
|--------|----------------------|---------|-------------|
| Full FT | 100% | Alta | Melhor |
| LoRA (r=8) | ~0.1% | Baixa | Próximo de full |
| QLoRA | ~0.1% | Muito Baixa | Próximo de full |

**LoRA Configuration:**
```python
from peft import LoraConfig
config = LoraConfig(
    r=16,               # Rank (8-64 típico)
    lora_alpha=16,      # Scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)
```

---

### 3. LLMs - Funcionamento e Técnicas

#### 3.1 Tokenização

| Algoritmo | Usado Por | Característica |
|-----------|-----------|----------------|
| **BPE** | GPT-2/3/4, LLaMA | Iterativo, byte-level |
| **WordPiece** | BERT | Maximiza likelihood |
| **SentencePiece** | T5, LLaMA | Language-agnostic |

**Regra de thumb**: ~4 caracteres ≈ 1 token (inglês); código tem densidade maior.

#### 3.2 Context Window Management

| Estratégia | Descrição | Quando Usar |
|------------|-----------|-------------|
| **Truncation** | Mantém tokens mais recentes | Conversas longas |
| **Sliding Window** | Overlap de 10-20% | Documentos grandes |
| **Summarization** | LLM resume histórico antigo | Contexto crítico |
| **RAG** | Recupera chunks relevantes | Base de conhecimento |

**Problema "Lost in the Middle"**: Informação no meio do contexto é frequentemente ignorada.
- **Solução**: Colocar informação crítica no início e fim.

#### 3.3 Prompt Engineering Avançado

**Chain of Thought (CoT):**
```
Resolva este problema. Vamos pensar passo a passo.
```

**ReAct Pattern:**
```
Thought: Preciso encontrar o clima em Paris.
Action: search["weather Paris today"]
Observation: Atualmente 18°C e nublado em Paris.
Thought: Agora posso responder.
Action: finish["Está 18°C e nublado em Paris."]
```

**Tree of Thoughts**: Para problemas que requerem backtracking.

#### 3.4 Function Calling (Tool Use)

```python
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Obtém clima atual para uma localização",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "Cidade e país"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}]
```

#### 3.5 Sampling Strategies

| Parâmetro | Efeito | Valores Típicos |
|-----------|--------|-----------------|
| **Temperature** | 0 = determinístico, >1 = criativo | 0.7-1.0 geral |
| **Top-p** | Nucleus sampling dinâmico | 0.9-0.95 |
| **Top-k** | Vocabulário fixo | 40-100 |

**Recomendação**: Tarefas factuais: T=0.2, top_p=0.9; Criativas: T=0.8, top_p=0.95.

---

## PARTE II: ARQUITETURA DE AGENTES AUTÔNOMOS

### 4. Arquitetura de Agentes (Core)

#### 4.1 Padrões Arquiteturais

**ReAct (Padrão Fundamental):**
```
Thought → Action → Observation → Thought → ... → Final Answer
```
- Performance: 12.29% no SWE-bench (baseline)
- Combina raciocínio verbal com ações

**Plan-and-Execute:**
```
┌─────────────────┐
│  PLANNER        │ → Gera plano multi-step
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EXECUTOR       │ → Executa cada step (modelo menor/especializado)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RE-PLANNER     │ → Avalia e ajusta plano
└─────────────────┘
```
- Mais rápido que ReAct (sub-tarefas não precisam de LLM call)
- Força "pensar" em todos os passos antes de agir

**LLMCompiler (Execução Paralela):**
- DAG de tarefas com dependências
- Task Fetching Unit schedula quando dependências satisfeitas
- **3.6x speedup** através de paralelismo

#### 4.2 Componentes Core do Agente

| Componente | Função | Implementação |
|------------|--------|---------------|
| **Planner** | Decomposição de tarefas, goal setting | LLM com structured output |
| **Executor** | Execução de ações, tool use | Sandbox + tool calls |
| **Critic** | Validação de output, self-correction | LLM-as-judge, testes |
| **Memory** | Contexto, aprendizado | Vector DB + working memory |
| **Perception** | Processamento de input | Embeddings, parsing |

#### 4.3 Sistemas de Memória

| Tipo | Descrição | Implementação |
|------|-----------|---------------|
| **Working Memory** | Contexto atual (context window) | LLM context |
| **Episodic Memory** | Experiências passadas | Vector DB com timestamps |
| **Semantic Memory** | Fatos e conceitos | RAG system |
| **Procedural Memory** | Skills, como-fazer | Tool definitions, templates |

**Padrão MemGPT/Letta:**
- **Core Memory**: Blocos in-context que agente gerencia
- **Recall Memory**: Histórico conversacional
- **Archival Memory**: DB externo para fatos

#### 4.4 State Machine do Agente

```
START
  │
  ▼
┌─────────────────┐
│ RECEIVE REQUEST │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PLAN GENERATION │◄────────────────┐
└────────┬────────┘                 │
         │                          │
         ▼                          │
┌─────────────────┐                 │
│ EXECUTE ACTION  │                 │
│ (in sandbox)    │                 │
└────────┬────────┘                 │
         │                          │
         ▼                          │
┌─────────────────┐     ┌───────────┴───────────┐
│ OBSERVE RESULT  │────►│ REPLAN IF NEEDED      │
└────────┬────────┘     └───────────────────────┘
         │
         ▼
┌─────────────────┐
│ MORE STEPS?     │───Yes───► (loop back)
└────────┬────────┘
         │ No
         ▼
┌─────────────────┐
│ VALIDATE OUTPUT │
│ (tests, review) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HUMAN APPROVAL? │───Yes───► WAIT FOR APPROVAL
└────────┬────────┘
         │ No/Approved
         ▼
       COMMIT & DEPLOY
```

#### 4.5 Multi-Agent Coordination

**Supervisor/Worker Pattern:**
```
[User Request] → [Supervisor] → [Coder Agent]
                              → [Reviewer Agent]
                              → [Tester Agent]
                              → [Docs Agent]
```

**Cursor 2.0**: Até 8 agentes paralelos com git worktree isolation.

#### 4.6 Approval Workflows

| Nível de Risco | Exemplos | Ação |
|----------------|----------|------|
| **Baixo** | Leitura, buscas | Auto-approve |
| **Médio** | Edição de arquivos, testes | Notificação |
| **Alto** | Git commits, deploys | Aprovação humana |
| **Crítico** | Produção, credentials | Multi-approval |

---

### 5. Execução Segura em Sandbox

#### 5.1 Tecnologias de Isolamento

| Tecnologia | Isolamento | Startup | Overhead | Melhor Para |
|------------|-----------|---------|----------|-------------|
| **Docker** (default) | Processo (namespaces) | ~100ms | ~10MB | Dev, código confiável |
| **gVisor** | User-space kernel | ~200ms | ~50MB | Código não-confiável |
| **Firecracker** | Hardware VM (KVM) | ~125ms | ~5MB | Produção, forte isolamento |
| **Kata Containers** | Hardware VM | ~500ms | ~128MB | Enterprise, compliance |
| **WebAssembly** | Sandboxed runtime | ~1ms | ~1MB | Edge, client-side |

**Recomendação Enterprise**: Firecracker microVMs (usado por AWS Lambda, E2B).

#### 5.2 Docker Hardening para Agentes

```yaml
version: '3.8'
services:
  ai-agent:
    image: ai-agent-sandbox:latest
    user: "10001:10001"           # Non-root
    read_only: true               # Filesystem read-only
    cap_drop: [ALL]               # Drop ALL capabilities
    security_opt:
      - no-new-privileges:true
      - seccomp:/etc/docker/seccomp/agent-profile.json
      - apparmor:docker-ai-agent
    tmpfs:
      - /tmp:size=100M,mode=1777,noexec
      - /workspace:size=500M,mode=1755,uid=10001
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
          pids: 100               # Previne fork bombs
    networks:
      - isolated
networks:
  isolated:
    internal: true                # Sem acesso externo
```

#### 5.3 Namespace/Cgroups Configuration

```bash
# Cgroups v2 - Limites de recursos
mkdir /sys/fs/cgroup/ai-agent
echo 536870912 > /sys/fs/cgroup/ai-agent/memory.max    # 512MB
echo "50000 100000" > /sys/fs/cgroup/ai-agent/cpu.max  # 50% CPU
echo 100 > /sys/fs/cgroup/ai-agent/pids.max            # 100 processos
```

#### 5.4 Seccomp Profile (Syscall Filtering)

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["read", "write", "open", "close", "mmap", "munmap", 
                "brk", "exit_group", "rt_sigaction", "rt_sigprocmask"],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

#### 5.5 E2B Cloud Sandbox (Produção)

```python
from e2b_code_interpreter import Sandbox

with Sandbox() as sandbox:
    # Executa código AI-gerado em microVM Firecracker
    result = sandbox.run_code("""
        import pandas as pd
        df = pd.DataFrame({'x': [1, 2, 3]})
        print(df.describe())
    """)
    # Operações de arquivo
    sandbox.filesystem.write('/workspace/output.txt', 'results')
    # Comandos de terminal
    sandbox.terminal.exec('pip install numpy')
```

---

### 6. RAG para Código

#### 6.1 Embedding Models Comparison

| Modelo | MTEB Score | Dimensões | Custo/1M | Melhor Para |
|--------|------------|-----------|----------|-------------|
| **Cohere embed-v4** | 65.2 | 1024 | $0.10 | Multilingual, Search |
| **OpenAI text-embedding-3-large** | 64.6 | 3072 | $0.13 | General purpose |
| **Voyage AI voyage-3** | 63.8 | 1024 | $0.06 | Domain-specific, RAG |
| **BGE-M3** (open-source) | 63.0 | 1024 | Free | Self-hosted, privacy |
| **voyage-code-3** | - | 1024 | $0.06 | **Código** |

#### 6.2 Vector Databases

| Database | Tipo | Escala | Melhor Para |
|----------|------|--------|-------------|
| **Pinecone** | Managed | Bilhões | Enterprise, hands-off |
| **Qdrant** | Open/Managed | Bilhões | Filtering complexo |
| **pgvector** | Extension | 10-100M | PostgreSQL users |
| **Weaviate** | Open/Managed | 100M+ | Knowledge graphs |

#### 6.3 Chunking para Código (AST-based)

**Tree-sitter para parsing language-agnostic:**
```python
import tree_sitter
parser = tree_sitter.Parser()
parser.set_language(tree_sitter_python.language())
tree = parser.parse(source_code)

# Extrair chunks semânticos
chunks = extract_functions(tree)   # Funções como chunks
chunks += extract_classes(tree)    # Classes como chunks
```

**Benefícios**: +5.5 pontos no RepoEval vs chunking textual.

#### 6.4 Retrieval Strategy

**Hybrid Search (Recomendado):**
```python
# Dense + Sparse = 15-30% melhor recall
hybrid_results = dense_results ∪ sparse_results
final_results = reciprocal_rank_fusion(hybrid_results)

# RRF Formula
RRF_score(d) = Σ 1/(60 + rank_r(d))
```

#### 6.5 Reranking (Essencial)

| Reranker | Tipo | Latência | Accuracy |
|----------|------|----------|----------|
| **Cohere Rerank** | API | ~100ms | +9.3pp no RAG |
| **Cross-encoder** | Model | ~100-500ms/50 docs | Highest |
| **ColBERT** | Late Interaction | Sub-segundo | Excelente |
| **FlashRank** | Lightweight | 80% menos latência | 95% de cross-encoder |

**Arquitetura Two-Stage:**
```
Stage 1: Retrieve top-100 (fast, recall-focused)
    ↓
Stage 2: Rerank to top-5 (slow, precision-focused)
```

#### 6.6 RAG Pipeline para Código

```
┌─────────────────────────────────────────────────────────────┐
│                     QUERY INPUT                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              QUERY PROCESSING                                │
│   • Query expansion (HyDE opcional)                         │
│   • Query routing (seleciona índice apropriado)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              PARALLEL RETRIEVAL                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│   │   Dense      │    │   Sparse     │    │   Code       │ │
│   │   (Vector)   │    │   (BM25)     │    │   (AST)      │ │
│   └──────────────┘    └──────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              FUSION (RRF) + RERANKING                        │
│   Merge results → top-50 → Cross-encoder → top-5            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              GENERATION                                      │
│   LLM + context + citations                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## PARTE III: INFRAESTRUTURA E BACKEND

### 7. Backend Architecture (NestJS)

#### 7.1 Estrutura de Módulos

```
src/
├── modules/
│   ├── ai/
│   │   ├── ai.module.ts
│   │   ├── services/
│   │   │   ├── llm.service.ts
│   │   │   ├── embedding.service.ts
│   │   │   └── prompt.service.ts
│   │   ├── processors/        # BullMQ workers
│   │   └── dto/
│   ├── conversation/
│   ├── vector-search/
│   └── agent/
├── infra/
│   ├── database/
│   ├── cache/
│   └── observability/
└── common/
    ├── decorators/
    ├── filters/
    ├── guards/
    └── interceptors/
```

#### 7.2 Prisma Schema para AI Apps

```prisma
model User {
  id            String         @id @default(cuid())
  conversations Conversation[]
  usage         UsageTracking[]
  tier          UserTier       @default(FREE)
}

enum UserTier { FREE PREMIUM ENTERPRISE }

model Conversation {
  id        String    @id @default(cuid())
  userId    String
  user      User      @relation(fields: [userId], references: [id])
  messages  Message[]
  metadata  Json?
  @@index([userId, createdAt])
}

model Message {
  id             String       @id @default(cuid())
  conversationId String
  role           MessageRole
  content        String       @db.Text
  tokenCount     Int?
  @@index([conversationId, createdAt])
}

enum MessageRole { USER ASSISTANT SYSTEM }

model UsageTracking {
  id         String   @id @default(cuid())
  userId     String
  tokensIn   Int
  tokensOut  Int
  model      String
  cost       Decimal  @db.Decimal(10, 6)
  @@index([userId, createdAt])
}
```

#### 7.3 Streaming com SSE (NestJS)

```typescript
@Controller('chat')
export class ChatController {
  @Sse('stream')
  @UseGuards(JwtAuthGuard, AIRateLimitGuard)
  streamChat(@Query('prompt') prompt: string): Observable<MessageEvent> {
    return new Observable((observer) => {
      this.llmService.streamCompletion(prompt)
        .then(async (stream) => {
          for await (const chunk of stream) {
            observer.next({ data: JSON.stringify({ content: chunk }) });
          }
          observer.complete();
        })
        .catch((error) => observer.error(error));
    });
  }
}
```

#### 7.4 BullMQ para Tarefas Assíncronas

```typescript
@Module({
  imports: [
    BullModule.forRoot({
      defaultJobOptions: {
        removeOnComplete: 100,
        removeOnFail: 1000,
        attempts: 3,
        backoff: { type: 'exponential', delay: 2000 },
      },
    }),
    BullModule.registerQueue(
      { name: 'ai-inference' },
      { name: 'embedding-generation' },
      { name: 'ai-inference-dlq' },  // Dead letter queue
    ),
  ],
})
export class QueueModule {}

@Processor('ai-inference')
export class AIProcessor extends WorkerHost {
  async process(job: Job<AIJobData>): Promise<AIJobResult> {
    await job.updateProgress(10);
    const result = await this.llmService.complete(job.data.prompt, {
      timeout: 120000,
    });
    await job.updateProgress(100);
    return { success: true, content: result };
  }
  
  @OnWorkerEvent('failed')
  async onFailed(job: Job, error: Error) {
    if (job.attemptsMade >= job.opts.attempts) {
      await this.dlqQueue.add('failed-job', {
        originalJob: job.data,
        error: error.message,
      });
    }
  }
}
```

#### 7.5 Exception Filter para Erros de LLM

```typescript
@Catch(LLMError)
export class LLMExceptionFilter implements ExceptionFilter {
  catch(exception: LLMError, host: ArgumentsHost) {
    const response = host.switchToHttp().getResponse();
    
    if (exception instanceof RateLimitError) {
      return response.status(429).json({
        error: 'rate_limit_exceeded',
        retryAfter: exception.retryAfter,
      });
    }
    
    if (exception instanceof ContextLengthError) {
      return response.status(400).json({
        error: 'context_too_long',
        maxTokens: exception.maxTokens,
      });
    }
    
    return response.status(503).json({
      error: 'llm_unavailable',
    });
  }
}
```

#### 7.6 Arquitetura de Escalabilidade

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │  API    │          │  API    │          │  API    │
   │ Server  │          │ Server  │          │ Server  │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
       ┌────▼────┐      ┌────▼────┐     ┌────▼────┐
       │ Redis   │      │  Redis  │     │  Redis  │
       │(Primary)│◄────►│(Replica)│     │ (Queue) │
       └─────────┘      └─────────┘     └────┬────┘
                                             │
                    ┌────────────────────────┼────────────┐
                    │                        │            │
               ┌────▼────┐             ┌────▼────┐  ┌────▼────┐
               │ Worker  │             │ Worker  │  │ Worker  │
               └─────────┘             └─────────┘  └─────────┘
                                             │
                             ┌───────────────┼───────────────┐
                             │               │               │
                        ┌────▼────┐    ┌─────▼─────┐  ┌─────▼─────┐
                        │PostgreSQL│    │PostgreSQL │  │  Vector   │
                        │ (Write)  │    │ (Read)    │  │    DB     │
                        └──────────┘    └───────────┘  └───────────┘
```

---

### 8. Frontend AI UX

#### 8.1 Streaming Implementation

**Vercel AI SDK (Padrão da Indústria):**
```typescript
// Server: streamText
import { streamText } from 'ai';
const result = await streamText({
  model: 'anthropic/claude-sonnet-4.5',
  prompt: 'Hello'
});
return result.toDataStreamResponse();

// Client: useChat hook
import { useChat } from 'ai/react';
const { messages, input, handleSubmit, isLoading, stop } = useChat();
```

#### 8.2 Code Editor Integration

**Monaco Editor para Diff:**
```typescript
import { DiffEditor } from '@monaco-editor/react';

<DiffEditor
  original={originalCode}
  modified={modifiedCode}
  language="typescript"
  options={{ renderSideBySide: true }}
/>
```

#### 8.3 Agent Timeline UI

| Elemento | Descrição |
|----------|-----------|
| **Status Indicators** | "Thinking...", "Searching...", "Executing..." |
| **Thought Log** | Raciocínio do AI visível |
| **Step-by-step** | Panels colapsáveis para cada step |
| **Tool Calls** | Nome, parâmetros, resultado |
| **Approval Buttons** | Accept/Reject para ações sensíveis |

#### 8.4 State Management

| Tipo de Estado | Ferramenta |
|----------------|------------|
| Server/API data | React Query |
| Global UI state | Zustand |
| Fine-grained local | Jotai |
| Form state | React Hook Form |

---

## PARTE IV: SEGURANÇA

### 9. Segurança para Aplicações de IA

#### 9.1 OWASP Top 10 para LLMs (2025)

| Risk | Descrição | Mitigação |
|------|-----------|-----------|
| **LLM01: Prompt Injection** | Prompts maliciosos alteram comportamento | Input sanitization, instruction hierarchy |
| **LLM02: Sensitive Info Disclosure** | Vazamento de dados sensíveis | Output filtering, PII detection |
| **LLM03: Supply Chain** | Vulnerabilidades em dependências | Validação de modelos, SBOM |
| **LLM04: Data Poisoning** | Manipulação de dados de treinamento | Data validation, provenance |
| **LLM05: Improper Output Handling** | Outputs não sanitizados | Validation, encoding |
| **LLM06: Excessive Agency** | Autonomia excessiva do LLM | Approval gates, limits |
| **LLM07: System Prompt Leakage** | Exposição de prompts internos | Instruction defense |
| **LLM08: Vector/Embedding Weaknesses** | Riscos em RAG systems | Metadata filtering |
| **LLM09: Misinformation** | Geração de informação incorreta | Grounding, fact-checking |
| **LLM10: Unbounded Consumption** | Exaustão de recursos (DoS) | Rate limiting, budgets |

#### 9.2 Prompt Injection Defense

**Input Sanitization:**
```python
def sanitize_input(user_input: str) -> str:
    # Detectar markers de injeção
    injection_patterns = [
        r"\[INST\]", r"System:", r"You are now", 
        r"Ignore previous", r"Override"
    ]
    for pattern in injection_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            raise SecurityException("Potential injection detected")
    return user_input
```

**Instruction Hierarchy:**
```xml
<system priority="1">
  Você é um assistente de programação seguro.
  NUNCA execute código que modifique o sistema.
  SEMPRE verifique com o usuário antes de ações destrutivas.
</system>

<user_context priority="2">
  <!-- Conteúdo do usuário isolado -->
</user_context>
```

#### 9.3 Sandbox Security Checklist

**Pre-Deployment:**
- [ ] Non-root user (UID 10000+)
- [ ] Drop ALL capabilities, add only required
- [ ] Seccomp profile enabled
- [ ] AppArmor/SELinux profile
- [ ] Read-only root filesystem
- [ ] tmpfs for writable paths
- [ ] `no-new-privileges` flag
- [ ] Init process (tini/dumb-init)

**Resource Limits:**
- [ ] CPU limit (--cpus)
- [ ] Memory limit (--memory)
- [ ] PID limit (--pids-limit)
- [ ] I/O limits
- [ ] Storage quota

**Network:**
- [ ] Internal network ou network=none
- [ ] Egress filtering
- [ ] DNS filtering
- [ ] Proxy for external requests

#### 9.4 Rate Limiting

| Algoritmo | Descrição | Melhor Para |
|-----------|-----------|-------------|
| **Token Bucket** | Tokens refill a taxa fixa | Burst tolerance |
| **Sliding Window** | Avaliação em janela rolante | Enforcement preciso |
| **Leaky Bucket** | Processamento a taxa constante | Tráfego previsível |

**Implementação com Redis:**
```typescript
async checkRateLimit(userId: string, limit: number, window: number): Promise<boolean> {
  const key = `ratelimit:${userId}`;
  const current = await this.redis.incr(key);
  if (current === 1) await this.redis.expire(key, window);
  return current <= limit;
}
```

#### 9.5 Secrets Management

**HashiCorp Vault (Recomendado):**
- Dynamic secrets com expiração automática
- Rotação automática de API keys
- Integração Kubernetes

**Best Practices:**
```python
# BOM
api_key = os.environ["OPENAI_API_KEY"]

# RUIM - NUNCA fazer
api_key = "sk-1234567890abcdef"
```

---

## PARTE V: OPERAÇÕES E MLOPS

### 10. MLOps para LLMs

#### 10.1 Model Versioning

| Ferramenta | Melhor Para |
|------------|-------------|
| **MLflow** | Open-source, registro centralizado |
| **Weights & Biases** | End-to-end ML lifecycle |
| **Langfuse** | Prompt management, observabilidade |

#### 10.2 Prompt Versioning

```python
# Langfuse - Labels para ambientes
prompt_prod = langfuse.get_prompt("my-prompt", label="production")
prompt_staging = langfuse.get_prompt("my-prompt", label="staging")
```

#### 10.3 Deployment Strategies

| Estratégia | Descrição | Quando Usar |
|------------|-----------|-------------|
| **Blue-Green** | Dois ambientes idênticos | Major updates |
| **Canary** | % pequeno de tráfego primeiro | Iterative updates |
| **Feature Flags** | Controle granular | A/B testing |

#### 10.4 Cost Optimization Playbook

**Immediate Wins (15-40% savings):**
- Prompt optimization (remover verbosidade)
- Response caching (15% hit rate exact, 42% semantic)
- Token budgeting

**Medium-term (30-50% savings):**
- Model routing (simple → cheap, complex → expensive)
- Batching (43x faster com vLLM)

**Long-term (60-85% savings):**
- Fine-tuning smaller models
- Quantization (INT8/INT4)
- Spot instances para batch processing

#### 10.5 Observability Stack

| Componente | Ferramenta |
|------------|------------|
| **Tracing** | OpenTelemetry + Langfuse |
| **Metrics** | Prometheus + Grafana |
| **Logging** | Pino + Loki |
| **LLM-specific** | Langfuse, W&B Weave |

**Métricas Chave:**
- Latency (TTFT, P50, P95, P99)
- Token usage (input vs output)
- Cost per request/user
- Hallucination rate
- Cache hit rate

---

## PARTE VI: AVALIAÇÃO E BENCHMARKS

### 11. LLM Evaluation

#### 11.1 Benchmarks

| Benchmark | Foco | Uso |
|-----------|------|-----|
| **MMLU** | Conhecimento geral | Baseline |
| **HumanEval** | Geração de código | Coding assistants |
| **SWE-bench** | Real-world coding | Agent evaluation |
| **MT-Bench** | Multi-turn | Chatbots |
| **Chatbot Arena** | Human preference | Gold standard |

#### 11.2 Hallucination Detection

| Método | Descrição |
|--------|-----------|
| **Semantic Entropy** | Detecta variação semântica entre amostras |
| **SelfCheckGPT** | Compara múltiplas respostas |
| **MetaQA** | Mutação de prompt + metamorphic testing |
| **RAG Faithfulness** | LLM-as-judge para RAG |

#### 11.3 LLM-as-a-Judge Best Practices

- Chain-of-thought reasoning (+10-15% reliability)
- Small integer scales (1-5)
- Few-shot examples (5-10)
- Temperature 0.2-0.3
- Self-consistency (3x runs)

#### 11.4 Safety Evaluation

| Benchmark | Foco |
|-----------|------|
| **HarmBench** | Red teaming padronizado |
| **JailbreakBench** | Resistência a jailbreak |
| **ToxicChat** | Detecção de conteúdo tóxico |
| **TruthfulQA** | Truthfulness |

---

## PARTE VII: TROUBLESHOOTING COMPLETO

### 12. Erros de Backend/Database

#### 12.1 Connection Pool Exhaustion

**Sintomas:**
- `SequelizeConnectionAcquireTimeoutError`
- Requests queuing
- Pool metrics: all connections "active"

**Diagnóstico:**
```sql
SELECT count(*) FROM pg_stat_activity;
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
```

**Solução:**
```javascript
const pool = new Pool({
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
});
// CRÍTICO: Sempre liberar conexões
async function query(sql) {
  const client = await pool.connect();
  try {
    return await client.query(sql);
  } finally {
    client.release(); // NUNCA esquecer
  }
}
```

#### 12.2 Deadlock Detection

**Diagnóstico PostgreSQL:**
```sql
SET log_lock_waits = on;
SELECT pid, state, wait_event, query 
FROM pg_stat_activity 
WHERE wait_event_type = 'Lock';
```

**Prevenção:**
1. Consistent lock ordering (alfabético ou por ID)
2. Transações curtas
3. `FOR NO KEY UPDATE` ao invés de `FOR UPDATE`

#### 12.3 Prisma Migration Recovery

```bash
# Marcar migração como rolled back
npx prisma migrate resolve --rolled-back "migration_name"

# Marcar como aplicada (após fix manual)
npx prisma migrate resolve --applied "migration_name"

# Gerar diff
npx prisma migrate diff --from-migrations ./migrations --to-schema-datamodel ./schema.prisma
```

#### 12.4 Memory Leaks (Node.js)

**Detecção:**
```bash
node --trace-gc app.js
node --inspect app.js  # Para heap snapshots
```

**Heap Snapshot Analysis:**
1. Abrir `chrome://inspect`
2. Tirar múltiplos snapshots em intervalos
3. Comparar snapshots: "Size Delta" column
4. Procurar: growing JSArrayBufferData, retained strings

**Causas Comuns:**
- Event listeners não removidos
- Closures capturando objetos grandes
- Variáveis globais acumulando
- Conexões/streams não fechadas

#### 12.5 Circuit Breaker Pattern

```javascript
const CircuitBreaker = require('opossum');

const options = {
  timeout: 3000,
  errorThresholdPercentage: 50,
  resetTimeout: 30000,
};

const breaker = new CircuitBreaker(asyncFunction, options);
breaker.fallback(() => getCachedData());

breaker.on('open', () => alertOps('Circuit opened!'));
breaker.on('close', () => console.log('Service recovered'));
```

---

### 13. Erros de LLM/Agent/RAG

#### 13.1 API Error Handling

| Erro | HTTP | Ação |
|------|------|------|
| **Rate Limit** | 429 | Exponential backoff, 5 retries |
| **Context Exceeded** | 400 | Truncate, summarize |
| **Timeout** | 504 | Retry com timeout maior |
| **Overloaded** | 529 | Backoff longo, fallback model |

**Retry with Exponential Backoff:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=60)
)
def robust_llm_call(**kwargs):
    return client.chat.completions.create(**kwargs)
```

#### 13.2 Agent Failure Modes

| Falha | Prevalência | Solução |
|-------|-------------|---------|
| **Infinite Loops** | Alta | `max_iterations` limit, cost cap |
| **Stuck States** | Média | Timeout + replanning |
| **Tool Call Errors** | Alta | Schema validation, fallback tools |
| **Context Exhaustion** | Alta | Summarization, memory offloading |

**Loop Detection:**
```python
class AgentExecutor:
    def __init__(self, max_iterations=10, cost_cap=1.0):
        self.max_iterations = max_iterations
        self.cost_cap = cost_cap
        self.action_history = []
    
    def detect_loop(self, action):
        # Detecta ações repetidas
        recent = self.action_history[-5:]
        if recent.count(action) >= 3:
            raise LoopDetectedError("Agent in loop")
        self.action_history.append(action)
```

#### 13.3 RAG Failure Diagnosis

```
┌───────────────────────────────────────┐
│ Problema: "I don't have information"  │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ Check 1: Retrieval scores             │
│ Scores < 0.5? → Improve chunking      │
└───────────────┬───────────────────────┘
                │ Scores OK
                ▼
┌───────────────────────────────────────┐
│ Check 2: Chunk content                │
│ Irrelevant? → Query expansion, HyDE   │
└───────────────┬───────────────────────┘
                │ Content OK
                ▼
┌───────────────────────────────────────┐
│ Check 3: Context position             │
│ Buried in middle? → Reduce K, rerank  │
└───────────────────────────────────────┘
```

#### 13.4 Streaming Failure Recovery

**Problema**: Connection drops durante streaming longo.

**Solução - Redis Streams:**
1. Decouple generation from delivery
2. Store chunks in Redis Stream
3. Client reconnects e resume from last position

```python
# Producer: gera para Redis
await redis.xadd('stream:session123', {'chunk': content, 'index': i})

# Consumer: resume de onde parou
last_id = get_last_consumed_id('session123')
chunks = await redis.xread({'stream:session123': last_id})
```

---

### 14. Runbooks de Troubleshooting

#### 14.1 Rate Limit (429) Runbook

**Ações Imediatas:**
1. Check usage atual vs tier limits
2. Verificar loops ou requests duplicados
3. Check se API key compartilhada

**Fix Curto Prazo:**
```python
@backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
def call_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)
```

**Fix Longo Prazo:**
- Request tier upgrade
- Implement request queuing
- Add caching
- Consider batch API

#### 14.2 Agent Infinite Loop Runbook

**Ações Imediatas:**
1. Terminate agente imediatamente
2. Review trace para start point do loop
3. Check termination conditions no prompt

**Fix:**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=15,  # Hard limit
    early_stopping_method="generate",  # Permite stop gracioso
)
```

#### 14.3 RAG Poor Retrieval Runbook

**Diagnóstico:**
```python
# 1. Inspecionar chunks recuperados
results = vector_store.similarity_search_with_score(query, k=10)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc.page_content[:100]}")

# 2. Verificar embedding
query_embedding = embedding_model.embed(query)
print(f"Query embedding stats: mean={np.mean(query_embedding):.3f}")
```

**Fixes por Problema:**

| Problema | Fix |
|----------|-----|
| Low Recall | Adicionar conteúdo ao índice |
| Low Precision | Melhorar chunking (semântico) |
| Lost in Middle | Reduzir K, adicionar reranking |
| Outdated | Implementar refresh schedule |

---

## PARTE VIII: PAGAMENTOS E COMPLIANCE

### 15. Stripe Integration

#### 15.1 Usage-Based Billing para Tokens

```python
# Stripe Billing Meters
curl https://api.stripe.com/v1/billing/meters \
  -d display_name="AI Tokens" \
  -d event_name=llm_tokens \
  -d "default_aggregation[formula]"=sum

# Reportar uso
stripe.billing.MeterEvent.create(
    event_name="llm_tokens",
    payload={
        "stripe_customer_id": customer.stripe_id,
        "value": token_count,
    }
)
```

#### 15.2 Pricing Tiers Recomendados

| Tier | Preço | Tokens/mês | Features |
|------|-------|------------|----------|
| **Free** | $0 | 10K | Basic models, 10 RPM |
| **Pro** | $29/mês | 500K | Advanced models, 100 RPM, API |
| **Enterprise** | Custom | Ilimitado | SLAs, SSO, dedicated support |

#### 15.3 Webhook Handling

```typescript
@Post('stripe-webhook')
async handleWebhook(@Req() req, @Res() res) {
  const event = stripe.webhooks.constructEvent(
    req.body,
    req.headers['stripe-signature'],
    process.env.STRIPE_WEBHOOK_SECRET
  );
  
  switch (event.type) {
    case 'invoice.payment_succeeded':
      await this.provisionAccess(event.data.object);
      break;
    case 'invoice.payment_failed':
      await this.triggerDunning(event.data.object);
      break;
    case 'customer.subscription.deleted':
      await this.revokeAccess(event.data.object);
      break;
  }
  
  res.status(200).send();
}
```

### 16. Compliance

#### 16.1 GDPR/LGPD Requirements

| Direito | Implementação |
|---------|---------------|
| **Acesso** | Data export endpoint (30 dias) |
| **Retificação** | Profile editing |
| **Apagamento** | Account deletion workflow |
| **Portabilidade** | JSON/CSV export |
| **Objeção** | Opt-out mechanisms |

#### 16.2 EU AI Act Compliance

| Nível de Risco | Exemplos | Requisitos |
|----------------|----------|------------|
| **Inaceitável** | Social scoring | Proibido |
| **Alto Risco** | Hiring, credit | Full compliance |
| **Risco Limitado** | Chatbots | Transparency |
| **Risco Mínimo** | Spam filters | Nenhum |

**Timeline:**
- Fev 2025: Prohibited practices
- Ago 2025: GPAI obligations
- Ago 2026: Full application

#### 16.3 Audit Logging

```json
{
  "timestamp": "ISO 8601 UTC",
  "request_id": "uuid",
  "actor": {
    "user_id": "hashed",
    "role": "admin|user|service",
    "ip_address": "anonymized"
  },
  "action": "api_call|login|data_export",
  "resource": "endpoint",
  "input": { "prompt_hash": "SHA-256" },
  "output": { "response_hash": "SHA-256" },
  "metadata": {
    "latency_ms": 150,
    "token_count": {"input": 500, "output": 1200}
  }
}
```

---

## PARTE IX: IMPLEMENTATION ROADMAP

### Fase 1: Foundation (Semanas 1-4)
- [ ] Setup ambiente de desenvolvimento
- [ ] Implementar single-agent ReAct loop
- [ ] Configurar sandbox básico (Docker hardened)
- [ ] Tools básicos: read, write, search, execute
- [ ] Git integration

### Fase 2: RAG & Memory (Semanas 5-8)
- [ ] Vector database (pgvector ou Qdrant)
- [ ] AST-based chunking para código
- [ ] Hybrid search (dense + BM25)
- [ ] Reranking com cross-encoder
- [ ] Working memory management

### Fase 3: Multi-Agent (Semanas 9-12)
- [ ] Supervisor-worker architecture
- [ ] Agents especializados (Coder, Reviewer, Tester)
- [ ] Protocolo de comunicação inter-agent
- [ ] Execução paralela com isolation

### Fase 4: Backend Production (Semanas 13-16)
- [ ] NestJS architecture completa
- [ ] BullMQ para async processing
- [ ] Redis caching layers
- [ ] PostgreSQL com Prisma
- [ ] Streaming endpoints (SSE)

### Fase 5: Security & Enterprise (Semanas 17-20)
- [ ] Security hardening completo
- [ ] Human approval workflows
- [ ] Audit logging
- [ ] SSO/SAML integration
- [ ] RBAC implementation

### Fase 6: MLOps & Observability (Semanas 21-24)
- [ ] OpenTelemetry tracing
- [ ] Langfuse para LLM observability
- [ ] Prometheus + Grafana dashboards
- [ ] A/B testing framework
- [ ] Cost monitoring

### Fase 7: Frontend & UX (Semanas 25-28)
- [ ] Chat interface com streaming
- [ ] Monaco editor integration
- [ ] Diff visualization
- [ ] Agent timeline UI
- [ ] Approval request UI

### Fase 8: Production Launch (Semanas 29-32)
- [ ] Load testing
- [ ] Security penetration testing
- [ ] Compliance audit
- [ ] Documentation completa
- [ ] Runbooks operacionais

---

## MÉTRICAS E KPIs POR COMPONENTE

| Componente | Métrica | Target | Alertar Se |
|------------|---------|--------|------------|
| **LLM API** | Latency P95 | <5s | >10s |
| **LLM API** | Error Rate | <1% | >5% |
| **Agent** | Task Completion | >80% | <60% |
| **Agent** | Loop Detection | 0 | >0 |
| **RAG** | Recall@5 | >0.8 | <0.7 |
| **RAG** | Faithfulness | >0.9 | <0.8 |
| **Backend** | Response Time P99 | <500ms | >1s |
| **Backend** | Availability | 99.9% | <99% |
| **Queue** | Job Failure Rate | <5% | >10% |
| **Queue** | Queue Depth | <1000 | >10000 |
| **Database** | Connection Pool Usage | <80% | >90% |
| **Cost** | Daily Spend | Budget | >120% budget |

---

## STACK TECNOLÓGICO RECOMENDADO

```yaml
Framework: NestJS 10+
Language: TypeScript 5.x
Runtime: Node.js 20 LTS

Database:
  Primary: PostgreSQL 16 + pgvector
  ORM: Prisma 6+
  
Caching:
  Layer: Redis 7+
  Client: ioredis
  
Queues:
  Library: BullMQ
  Dashboard: Bull Board
  
LLM:
  Primary: Claude 4 Sonnet
  Fallback: GPT-4o
  Embeddings: voyage-code-3 ou text-embedding-3-large
  
Vector DB:
  Development: Chroma
  Production: Qdrant ou Pinecone
  
Sandbox:
  Development: Docker hardened
  Production: Firecracker (E2B)
  
Observability:
  Tracing: OpenTelemetry + Langfuse
  Metrics: Prometheus + Grafana
  Logging: Pino + Loki
  
Frontend:
  Framework: Next.js 15
  UI: Shadcn/ui
  State: Zustand + React Query
  Editor: Monaco Editor
  AI SDK: Vercel AI SDK 6
  
Infrastructure:
  Container: Docker
  Orchestration: Kubernetes
  CI/CD: GitHub Actions
  Cloud: AWS/GCP/Azure
```

---

## CONCLUSÃO

Este plano enterprise fornece uma base completa para construir um assistente de codificação de IA de classe mundial. Os elementos críticos de sucesso são:

1. **Arquitetura de Agentes Robusta**: Plan-and-Execute com multi-agent coordination
2. **Sandbox Seguro**: Firecracker microVMs para execução de código não-confiável
3. **RAG Otimizado**: Hybrid search + AST-based chunking + cross-encoder reranking
4. **Observabilidade Profunda**: Traces de ponta-a-ponta para cada interação
5. **Troubleshooting Proativo**: Runbooks e alertas para todas as failure modes

Os líderes do mercado (Cursor, Devin, Claude Code) demonstram que a combinação destes elementos, implementados com rigor, resulta em sistemas que efetivamente multiplicam a produtividade dos desenvolvedores enquanto mantêm segurança e confiabilidade enterprise-grade.
