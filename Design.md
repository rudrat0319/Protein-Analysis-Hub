# Design Document: Protein Analysis Hub

## Overview

The Protein Analysis Hub is a web-based platform that provides researchers and clinicians with an AI-powered reasoning interface for understanding protein behavior, stability, and structural implications. The system integrates with external computational biology tools (AlphaFold, Rosetta), accepts various data uploads (PDB structures, sequences, documents, images), and uses open-source AI models to synthesize fragmented information into coherent, human-readable explanations.

The platform addresses the gap between structure prediction (now solved) and understanding WHY proteins behave certain ways. It highlights uncertainty, identifies conflicting evidence, and provides a project-centric unified interface for making sense of findings from multiple sources.

**Core Principles:**
- Helps interpret protein data (does NOT replace experiments)
- Integrates existing tools (does NOT guarantee biological outcomes)
- Assists in research with explicit uncertainty tracking

## Technology Stack

### Frontend
- **Framework**: Next.js 14+ with React 18+ and TypeScript
- **Styling**: Tailwind CSS for rapid UI development
- **3D Visualization**: Mol* (Molstar) for protein structure viewing
- **State Management**: React Context API / Zustand
- **API Communication**: Fetch API / Axios

### Backend
- **Framework**: FastAPI (Python 3.10+)
- **Server**: Uvicorn (ASGI server)
- **Database**: PostgreSQL 15+
- **ORM**: SQLAlchemy 2.0+
- **File Storage**: Local filesystem / S3-compatible storage

### File Handling & Parsing
- **Biopython**: Protein structure and sequence parsing
- **pandas/numpy**: Data manipulation and analysis
- **MDAnalysis**: (Optional) Molecular dynamics trajectory analysis
- **PyPDF2/pdfplumber**: PDF text extraction
- **openpyxl**: Excel file parsing

### Search & Retrieval (RAG)
- **FAISS**: Vector similarity search
- **SentenceTransformers**: Text embedding generation
- **LangChain/LlamaIndex**: RAG orchestration and tool calling

### AI Reasoning Stack (Open Source)
- **Ollama**: Local LLM serving
  - Models: Llama 3 8B, Mistral 7B, Qwen2.5 7B
- **Alternatives**: llama.cpp, vLLM for production scaling
- **Frameworks**: 
  - Hugging Face Transformers
  - PEFT (LoRA/QLoRA) for fine-tuning
  - Bitsandbytes for quantization

### Compute
- **Development**: Kaggle Notebooks (free GPU) or Google Colab
- **Production**: Cloud GPU instances (AWS, GCP, or on-premise)

## Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│              Web Interface (Next.js + React)            │
│  - Project Dashboard                                    │
│  - Chat Interface                                       │
│  - Notes Panel                                          │
│  - 3D Structure Viewer (Mol*)                          │
│  - File Upload                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓ (REST API)
┌─────────────────────────────────────────────────────────┐
│                   API Gateway (FastAPI)                 │
│  - Authentication & Authorization                       │
│  - Request Validation                                   │
│  - Rate Limiting                                        │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Project    │  │     Data     │  │  AI Reasoning│
│  Management  │  │  Management  │  │    Engine    │
│              │  │              │  │   (Ollama)   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        │                 │                 ↓
        │                 │         ┌──────────────┐
        │                 │         │ RAG Pipeline │
        │                 │         │  (LangChain) │
        │                 │         └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              External Tool Integration Layer            │
│  - AlphaFold Adapter                                    │
│  - Rosetta Adapter                                      │
│  - Other Tool Adapters                                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Persistence Layer                      │
│  - PostgreSQL (metadata, projects, chats, notes)       │
│  - File Storage (artifacts, structures, exports)       │
│  - FAISS Index (vector embeddings)                     │
└─────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Project-centric organization**: All work organized in Projects (not isolated chats)
2. **Async processing**: AI reasoning and external tool calls handled asynchronously
3. **RAG-based reasoning**: Use retrieval-augmented generation for context-aware responses
4. **Pluggable tool adapters**: External tools integrated via adapter pattern
5. **Editable AI outputs**: All AI responses stored separately from user edits
6. **Local-first AI**: Open-source models run locally via Ollama for privacy and cost

## Components and Interfaces

### 1. Project Management Component

**Responsibility**: Manage projects, including creation, persistence, retrieval, and deletion. Projects are the top-level organizational unit.

**Interface**:
```typescript
interface ProjectManager {
  createProject(userId: string, name: string, description: string): Promise<Project>
  getProject(projectId: string): Promise<Project>
  listProjects(userId: string): Promise<Project[]>
  updateProject(projectId: string, updates: Partial<Project>): Promise<Project>
  deleteProject(projectId: string): Promise<void>
  archiveProject(projectId: string): Promise<void>
  linkProjects(projectId1: string, projectId2: string, linkType: 'full' | 'partial'): Promise<void>
  mergeProjects(projectIds: string[]): Promise<Project>
}

interface Project {
  id: string
  userId: string
  name: string
  description: string
  createdAt: Date
  updatedAt: Date
  artifactIds: string[]
  chatHistory: ChatMessage[]
  notes: ResearchNote[]
  hypotheses: Hypothesis[]
  status: 'active' | 'archived'
  linkedProjects: string[]
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  artifactReferences: string[]
  edited: boolean
  originalContent?: string
  editedBy?: string
  editedAt?: Date
}

interface ResearchNote {
  id: string
  content: string
  createdAt: Date
  updatedAt: Date
  linkedChatMessages: string[]
  linkedArtifacts: string[]
  linkedFindings: string[]
  createdBy: string
}

interface Hypothesis {
  id: string
  description: string
  status: 'proposed' | 'testing' | 'confirmed' | 'rejected'
  createdAt: Date
  updatedAt: Date
  linkedData: string[]
  decisions: Decision[]
  notes: string[]
}

interface Decision {
  id: string
  description: string
  status: 'pending' | 'approved' | 'rejected'
  timestamp: Date
  rationale: string
}
```

### 2. Chat and Reasoning Component

**Responsibility**: Handle chat interactions, invoke AI reasoning engine, manage editable AI outputs.

**Interface**:
```typescript
interface ChatManager {
  sendMessage(projectId: string, message: string): Promise<ChatMessage>
  getChatHistory(projectId: string): Promise<ChatMessage[]>
  editAIOutput(messageId: string, newContent: string, userId: string): Promise<ChatMessage>
  revertEdit(messageId: string): Promise<ChatMessage>
  addAnnotation(messageId: string, annotation: Annotation): Promise<void>
}

interface Annotation {
  id: string
  messageId: string
  type: 'text' | 'structure' | 'diagram'
  content: string
  position?: { x: number; y: number }
  createdBy: string
  createdAt: Date
}
```

### 2. Data Management Component

**Responsibility**: Handle upload, validation, storage, and retrieval of data artifacts. Parse protein structures and sequences.

**Interface**:
```typescript
interface DataManager {
  uploadArtifact(file: File, projectId: string): Promise<Artifact>
  validateArtifact(file: File): Promise<ValidationResult>
  getArtifact(artifactId: string): Promise<Artifact>
  deleteArtifact(artifactId: string): Promise<void>
  parseArtifact(artifactId: string): Promise<ParsedData>
  parseProteinStructure(artifactId: string): Promise<ProteinStructure>
  parseSequence(artifactId: string): Promise<Sequence>
}

interface Artifact {
  id: string
  projectId: string
  fileName: string
  fileType: FileType
  fileSize: number
  uploadedAt: Date
  storageUrl: string
  metadata: Record<string, any>
}

type FileType = 
  | 'pdb' | 'mmcif' | 'fasta' | 'genbank'  // Structure & sequence
  | 'pdf' | 'docx' | 'txt'                  // Documents
  | 'csv' | 'xlsx' | 'json'                 // Data
  | 'png' | 'jpg' | 'tiff'                  // Images

interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

interface ParsedData {
  artifactId: string
  contentType: string
  extractedText?: string
  structuredData?: any
  metadata: Record<string, any>
}

interface ProteinStructure {
  artifactId: string
  pdbId?: string
  chains: Chain[]
  residues: Residue[]
  atoms: Atom[]
  metadata: {
    resolution?: number
    method?: string
    organism?: string
  }
}

interface Chain {
  id: string
  sequence: string
  residues: Residue[]
}

interface Residue {
  id: number
  name: string
  chainId: string
  atoms: Atom[]
}

interface Atom {
  id: number
  element: string
  x: number
  y: number
  z: number
  residueId: number
}

interface Sequence {
  artifactId: string
  id: string
  sequence: string
  length: number
  type: 'protein' | 'dna' | 'rna'
  metadata: Record<string, any>
}
```

### 3. 3D Visualization Component

**Responsibility**: Render protein structures in 3D using Mol*, handle user interactions and annotations.

**Interface**:
```typescript
interface StructureViewer {
  loadStructure(artifactId: string): Promise<void>
  setRepresentation(style: RepresentationStyle): void
  highlightResidues(residueIds: number[]): void
  addAnnotation(annotation: StructureAnnotation): void
  getAnnotations(): StructureAnnotation[]
  exportImage(format: 'png' | 'svg', resolution: number): Promise<Blob>
  saveViewState(): ViewState
  loadViewState(state: ViewState): void
}

type RepresentationStyle = 'cartoon' | 'surface' | 'stick' | 'ball-and-stick' | 'ribbon'

interface StructureAnnotation {
  id: string
  type: 'label' | 'arrow' | 'highlight' | 'measurement'
  residueIds: number[]
  text?: string
  color?: string
  createdBy: string
  createdAt: Date
}

interface ViewState {
  cameraPosition: { x: number; y: number; z: number }
  cameraRotation: { x: number; y: number; z: number }
  zoom: number
  representation: RepresentationStyle
  annotations: StructureAnnotation[]
}
```

### 4. AI Reasoning Engine (Ollama + RAG)

**Responsibility**: Synthesize information from multiple sources, generate explanations, identify conflicts and uncertainty. Use RAG for context-aware responses.

**Implementation**: Python backend using Ollama for LLM inference, LangChain for RAG orchestration, FAISS for vector search.

**Interface**:
```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ConfidenceLevel(Enum):
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"        # < 0.5

@dataclass
class DataSource:
    type: str  # 'artifact' | 'tool_result'
    id: str
    content: Any
    metadata: Dict[str, Any]

@dataclass
class Evidence:
    source_id: str
    source_type: str
    content: str
    relevance: float
    confidence: float

@dataclass
class Conclusion:
    statement: str
    confidence: float
    supporting_evidence: List[Evidence]
    conclusion_type: str  # 'established_fact' | 'inference' | 'hypothesis'
    uncertainty_factors: List[str]

@dataclass
class Conflict:
    description: str
    conflicting_sources: List[str]
    severity: str  # 'low' | 'medium' | 'high'
    resolution: Optional[str] = None

@dataclass
class AnalysisResult:
    project_id: str
    summary: str
    key_findings: List[Dict[str, Any]]
    conclusions: List[Conclusion]
    conflicts: List[Conflict]
    recommendations: List[str]
    generated_at: str

class AIReasoningEngine:
    def __init__(self, model_name: str = "llama3:8b"):
        """Initialize with Ollama model"""
        self.model_name = model_name
        self.rag_pipeline = self._init_rag_pipeline()
    
    def _init_rag_pipeline(self):
        """Initialize RAG pipeline with LangChain and FAISS"""
        pass
    
    async def analyze_project(self, project_id: str) -> AnalysisResult:
        """Analyze all data in a project and generate comprehensive analysis"""
        pass
    
    async def chat_response(
        self, 
        project_id: str, 
        message: str, 
        context: List[DataSource]
    ) -> str:
        """Generate chat response using RAG"""
        pass
    
    async def synthesize_information(
        self, 
        sources: List[DataSource]
    ) -> Dict[str, Any]:
        """Synthesize information from multiple sources"""
        pass
    
    async def explain_protein_behavior(
        self, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Explain protein behavior based on context"""
        pass
    
    async def identify_conflicts(
        self, 
        sources: List[DataSource]
    ) -> List[Conflict]:
        """Identify conflicting information across sources"""
        pass
    
    async def assess_uncertainty(
        self, 
        conclusion: str, 
        evidence: List[Evidence]
    ) -> Dict[str, Any]:
        """Assess uncertainty and confidence for a conclusion"""
        pass
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using SentenceTransformers"""
        pass
    
    async def search_similar(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar content using FAISS"""
        pass
```

### 5. External Tool Integration Layer

**Responsibility**: Connect with and orchestrate calls to external computational biology tools (AlphaFold, Rosetta, etc.).

**Interface**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ToolConnection:
    tool_name: str
    connected: bool
    connection_id: str
    expires_at: Optional[str] = None

@dataclass
class ToolResult:
    tool_name: str
    request_id: str
    status: str  # 'success' | 'error' | 'timeout'
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class ToolAdapter(ABC):
    """Abstract base class for tool adapters"""
    
    @abstractmethod
    async def connect(self, credentials: Dict[str, Any]) -> ToolConnection:
        """Establish connection to external tool"""
        pass
    
    @abstractmethod
    async def invoke(self, params: Dict[str, Any]) -> ToolResult:
        """Invoke the tool with given parameters"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the tool"""
        pass

class AlphaFoldAdapter(ToolAdapter):
    """Adapter for AlphaFold structure prediction"""
    
    async def connect(self, credentials: Dict[str, Any]) -> ToolConnection:
        # Implementation for AlphaFold connection
        pass
    
    async def invoke(self, params: Dict[str, Any]) -> ToolResult:
        # params: { 'sequence': str, 'model_preset': str, ... }
        pass
    
    async def disconnect(self) -> None:
        pass

class RosettaAdapter(ToolAdapter):
    """Adapter for Rosetta protein modeling"""
    
    async def connect(self, credentials: Dict[str, Any]) -> ToolConnection:
        pass
    
    async def invoke(self, params: Dict[str, Any]) -> ToolResult:
        # params: { 'pdb_file': str, 'protocol': str, ... }
        pass
    
    async def disconnect(self) -> None:
        pass

class ToolIntegrationManager:
    """Manages all external tool integrations"""
    
    def __init__(self):
        self.adapters: Dict[str, ToolAdapter] = {}
    
    def register_tool(self, tool_name: str, adapter: ToolAdapter):
        """Register a new tool adapter"""
        self.adapters[tool_name] = adapter
    
    async def connect_tool(
        self, 
        tool_name: str, 
        credentials: Dict[str, Any]
    ) -> ToolConnection:
        """Connect to a specific tool"""
        adapter = self.adapters.get(tool_name)
        if not adapter:
            raise ValueError(f"Tool {tool_name} not registered")
        return await adapter.connect(credentials)
    
    async def invoke_tool(
        self, 
        tool_name: str, 
        params: Dict[str, Any]
    ) -> ToolResult:
        """Invoke a specific tool"""
        adapter = self.adapters.get(tool_name)
        if not adapter:
            raise ValueError(f"Tool {tool_name} not registered")
        return await adapter.invoke(params)
    
    async def get_tool_status(self, tool_name: str) -> Dict[str, Any]:
        """Get status of a specific tool"""
        pass
```

### 6. Export and Report Generation Component

**Responsibility**: Generate and export project summaries, annotated diagrams, and reasoning summaries in various formats.

**Interface**:
```python
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ExportFormat(Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    PPTX = "pptx"
    HTML = "html"

@dataclass
class ExportOptions:
    format: ExportFormat
    include_chat_history: bool = True
    include_annotations: bool = True
    include_citations: bool = True
    include_structures: bool = True
    high_resolution: bool = False

class ReportGenerator:
    """Generate and export project reports"""
    
    async def generate_project_summary(
        self, 
        project_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive project summary"""
        pass
    
    async def export_report(
        self, 
        project_id: str, 
        options: ExportOptions
    ) -> bytes:
        """Export project report in specified format"""
        pass
    
    async def export_structure_image(
        self, 
        artifact_id: str, 
        view_state: Dict[str, Any],
        resolution: int = 300
    ) -> bytes:
        """Export annotated structure image"""
        pass
    
    async def export_chat_conversation(
        self, 
        project_id: str, 
        format: ExportFormat
    ) -> bytes:
        """Export chat conversation"""
        pass
    
    def _format_citations(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations for export"""
        pass
    
    def _generate_pdf(self, content: Dict[str, Any]) -> bytes:
        """Generate PDF from content"""
        pass
    
    def _generate_markdown(self, content: Dict[str, Any]) -> str:
        """Generate Markdown from content"""
        pass
    
    def _generate_pptx(self, content: Dict[str, Any]) -> bytes:
        """Generate PowerPoint from content"""
        pass
```

## Data Models

### Database Schema (PostgreSQL)

**Users Table**:
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    institution VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP
);
```

**Projects Table**:
```sql
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    CONSTRAINT status_check CHECK (status IN ('active', 'archived'))
);
```

**Project Links Table**:
```sql
CREATE TABLE project_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id_1 UUID REFERENCES projects(id) ON DELETE CASCADE,
    project_id_2 UUID REFERENCES projects(id) ON DELETE CASCADE,
    link_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT link_type_check CHECK (link_type IN ('full', 'partial'))
);
```

**Artifacts Table**:
```sql
CREATE TABLE artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    storage_url TEXT NOT NULL,
    metadata JSONB,
    parsed_data JSONB
);
```

**Chat Messages Table**:
```sql
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    artifact_references UUID[],
    edited BOOLEAN DEFAULT FALSE,
    original_content TEXT,
    edited_by UUID REFERENCES users(id),
    edited_at TIMESTAMP,
    CONSTRAINT role_check CHECK (role IN ('user', 'assistant'))
);
```

**Research Notes Table**:
```sql
CREATE TABLE research_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id),
    linked_chat_messages UUID[],
    linked_artifacts UUID[],
    linked_findings TEXT[]
);
```

**Hypotheses Table**:
```sql
CREATE TABLE hypotheses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'proposed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    linked_data UUID[],
    notes TEXT[],
    CONSTRAINT status_check CHECK (status IN ('proposed', 'testing', 'confirmed', 'rejected'))
);
```

**Decisions Table**:
```sql
CREATE TABLE decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id UUID REFERENCES hypotheses(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rationale TEXT,
    CONSTRAINT status_check CHECK (status IN ('pending', 'approved', 'rejected'))
);
```

**Tool Results Table**:
```sql
CREATE TABLE tool_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    tool_name VARCHAR(100) NOT NULL,
    request_params JSONB,
    result_data JSONB,
    status VARCHAR(50) NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_time FLOAT
);
```

**Structure Annotations Table**:
```sql
CREATE TABLE structure_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID REFERENCES artifacts(id) ON DELETE CASCADE,
    annotation_type VARCHAR(50) NOT NULL,
    residue_ids INTEGER[],
    text TEXT,
    color VARCHAR(50),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);
```

### File Storage Structure

```
/storage
  /users
    /{user_id}
      /projects
        /{project_id}
          /artifacts
            /{artifact_id}.{ext}
          /exports
            /{export_id}.{format}
          /tool-results
            /{tool_result_id}.json
          /structure-images
            /{image_id}.png
```

### Vector Store (FAISS)

**Embeddings Index**:
- Store embeddings for all text content (chat messages, notes, artifact text)
- Use SentenceTransformers (all-MiniLM-L6-v2 or similar)
- Dimension: 384 or 768 depending on model
- Index type: IndexFlatL2 for small datasets, IndexIVFFlat for larger

**Metadata Store**:
```python
{
    "id": "uuid",
    "project_id": "uuid",
    "content_type": "chat_message | note | artifact",
    "content": "text content",
    "timestamp": "iso8601",
    "metadata": {}
}
```


## Implementation Details

### Frontend Implementation (Next.js + React + TypeScript)

**Project Structure**:
```
/frontend
  /app
    /page.tsx                    # Landing page
    /dashboard
      /page.tsx                  # Project dashboard
    /project/[id]
      /page.tsx                  # Project workspace
      /layout.tsx                # Project layout
  /components
    /chat
      /ChatInterface.tsx         # Chat UI
      /MessageList.tsx           # Message display
      /MessageInput.tsx          # Input field
    /notes
      /NotesPanel.tsx            # Notes sidebar
      /NoteEditor.tsx            # Note editing
    /structure
      /StructureViewer.tsx       # Mol* integration
      /AnnotationTools.tsx       # Annotation UI
    /upload
      /FileUpload.tsx            # File upload component
      /FileList.tsx              # Uploaded files list
    /hypothesis
      /HypothesisTracker.tsx     # Hypothesis management
  /lib
    /api.ts                      # API client
    /types.ts                    # TypeScript types
    /utils.ts                    # Utility functions
  /styles
    /globals.css                 # Global styles (Tailwind)
```

**Key Components**:

1. **ChatInterface**: Real-time chat with AI, supports file references, editable outputs
2. **StructureViewer**: Mol* wrapper for 3D protein visualization
3. **NotesPanel**: Sidebar for research notes with linking capabilities
4. **HypothesisTracker**: UI for managing hypotheses and decisions

### Backend Implementation (FastAPI + Python)

**Project Structure**:
```
/backend
  /app
    /main.py                     # FastAPI app entry point
    /config.py                   # Configuration
    /database.py                 # Database connection
    /models
      /user.py                   # User model
      /project.py                # Project model
      /artifact.py               # Artifact model
      /chat.py                   # Chat message model
      /note.py                   # Research note model
      /hypothesis.py             # Hypothesis model
    /schemas
      /project.py                # Pydantic schemas
      /chat.py
      /artifact.py
    /api
      /routes
        /projects.py             # Project endpoints
        /chat.py                 # Chat endpoints
        /artifacts.py            # File upload endpoints
        /tools.py                # External tool endpoints
        /export.py               # Export endpoints
    /services
      /ai_reasoning.py           # AI reasoning engine
      /rag_pipeline.py           # RAG implementation
      /tool_integration.py       # Tool adapters
      /file_parser.py            # File parsing (Biopython)
      /export_service.py         # Export generation
    /utils
      /embeddings.py             # SentenceTransformers
      /vector_store.py           # FAISS operations
  /tests
    /unit                        # Unit tests
    /integration                 # Integration tests
    /property                    # Property-based tests
```

**Key Services**:

1. **AIReasoningEngine**: Ollama integration, prompt engineering, RAG
2. **RAGPipeline**: LangChain orchestration, FAISS search, context retrieval
3. **ToolIntegrationManager**: AlphaFold/Rosetta adapters
4. **FileParser**: Biopython for PDB/FASTA, pandas for CSV/Excel

### AI Reasoning Implementation

**Ollama Setup**:
```python
# Install Ollama and pull models
# ollama pull llama3:8b
# ollama pull mistral:7b

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class AIReasoningEngine:
    def __init__(self, model_name="llama3:8b"):
        self.llm = Ollama(model=model_name, temperature=0.7)
        self.rag_pipeline = RAGPipeline()
    
    async def chat_response(self, project_id: str, message: str):
        # Retrieve relevant context from FAISS
        context = await self.rag_pipeline.retrieve_context(
            project_id, message, top_k=5
        )
        
        # Build prompt with context
        prompt = self._build_prompt(message, context)
        
        # Generate response
        response = await self.llm.agenerate([prompt])
        
        # Extract evidence and confidence
        parsed = self._parse_response(response)
        
        return parsed
    
    def _build_prompt(self, message: str, context: List[Dict]) -> str:
        template = """You are a protein analysis expert assistant. 
        
Context from project:
{context}

User question: {message}

Provide a detailed, scientifically accurate response. Include:
1. Clear explanation in human-readable language
2. Evidence citations (reference specific data sources)
3. Confidence level (high/medium/low)
4. Uncertainty factors if applicable
5. Recommendations for additional data if needed

Response:"""
        
        context_str = "\n".join([
            f"- {c['content_type']}: {c['content'][:200]}..."
            for c in context
        ])
        
        return template.format(context=context_str, message=message)
```

**RAG Pipeline**:
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_stores = {}  # project_id -> FAISS index
    
    async def index_content(self, project_id: str, content: Dict):
        """Add content to vector store"""
        if project_id not in self.vector_stores:
            self.vector_stores[project_id] = FAISS.from_texts(
                [], self.embeddings
            )
        
        text = content['content']
        metadata = {
            'id': content['id'],
            'type': content['content_type'],
            'timestamp': content['timestamp']
        }
        
        self.vector_stores[project_id].add_texts(
            [text], metadatas=[metadata]
        )
    
    async def retrieve_context(
        self, 
        project_id: str, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve relevant context for query"""
        if project_id not in self.vector_stores:
            return []
        
        results = self.vector_stores[project_id].similarity_search(
            query, k=top_k
        )
        
        return [
            {
                'content': doc.page_content,
                'content_type': doc.metadata['type'],
                'id': doc.metadata['id']
            }
            for doc in results
        ]
```

### External Tool Integration

**AlphaFold Adapter Example**:
```python
import requests
from typing import Dict, Any

class AlphaFoldAdapter(ToolAdapter):
    def __init__(self, api_url: str = "https://alphafold.ebi.ac.uk/api"):
        self.api_url = api_url
        self.connected = False
    
    async def connect(self, credentials: Dict[str, Any]) -> ToolConnection:
        # Test connection
        try:
            response = requests.get(f"{self.api_url}/prediction")
            self.connected = response.status_code == 200
            return ToolConnection(
                tool_name="alphafold",
                connected=self.connected,
                connection_id="alphafold_conn_1"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to AlphaFold: {e}")
    
    async def invoke(self, params: Dict[str, Any]) -> ToolResult:
        """
        Fetch AlphaFold prediction
        params: { 'uniprot_id': str } or { 'sequence': str }
        """
        start_time = time.time()
        
        try:
            if 'uniprot_id' in params:
                # Fetch existing prediction
                url = f"{self.api_url}/prediction/{params['uniprot_id']}"
                response = requests.get(url)
                data = response.json()
            else:
                # Submit new prediction (if API supports)
                # This is simplified - actual implementation depends on API
                data = {"error": "Custom predictions not supported"}
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name="alphafold",
                request_id=str(uuid.uuid4()),
                status="success",
                data=data,
                execution_time=execution_time,
                metadata=params
            )
        except Exception as e:
            return ToolResult(
                tool_name="alphafold",
                request_id=str(uuid.uuid4()),
                status="error",
                error=str(e),
                execution_time=time.time() - start_time
            )
```

### File Parsing with Biopython

```python
from Bio.PDB import PDBParser, MMCIFParser
from Bio import SeqIO
import pandas as pd

class FileParser:
    @staticmethod
    async def parse_pdb(file_path: str) -> ProteinStructure:
        """Parse PDB file"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', file_path)
        
        chains = []
        for model in structure:
            for chain in model:
                residues = []
                for residue in chain:
                    atoms = []
                    for atom in residue:
                        atoms.append({
                            'element': atom.element,
                            'x': atom.coord[0],
                            'y': atom.coord[1],
                            'z': atom.coord[2]
                        })
                    residues.append({
                        'id': residue.id[1],
                        'name': residue.resname,
                        'atoms': atoms
                    })
                chains.append({
                    'id': chain.id,
                    'residues': residues
                })
        
        return ProteinStructure(chains=chains)
    
    @staticmethod
    async def parse_fasta(file_path: str) -> List[Sequence]:
        """Parse FASTA file"""
        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(Sequence(
                id=record.id,
                sequence=str(record.seq),
                length=len(record.seq),
                type='protein'
            ))
        return sequences
    
    @staticmethod
    async def parse_csv(file_path: str) -> pd.DataFrame:
        """Parse CSV file"""
        return pd.read_csv(file_path)
```


## Error Handling

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: File Upload and Validation

*For any* file upload attempt, the system should validate both the file format and size, accepting valid files under 100MB and rejecting others with appropriate error messages.

**Validates: Requirements 1.8, 1.9**

### Property 2: Chat Response Timeliness

*For any* user message sent in the chat interface, the AI reasoning engine should respond within 30 seconds.

**Validates: Requirements 1.11**

### Property 3: Chat History Persistence

*For any* project, all chat messages should be persisted and retrievable when the project is reopened.

**Validates: Requirements 1.12**

### Property 4: Tool Result Integration

*For any* external tool result received, it should be imported into the project and made available for AI reasoning.

**Validates: Requirements 2.5**

### Property 5: Tool Connection Error Handling

*For any* external tool connection failure, the system should both log the error with details and send a descriptive notification to the user.

**Validates: Requirements 2.8**

### Property 6: Research Note Linking

*For any* research note created, it should be correctly associated with the current project and any linked chat messages, artifacts, or findings.

**Validates: Requirements 3.3, 3.4, 3.5, 3.6**

### Property 7: Research Note Timestamps

*For any* research note, it should have accurate creation and modification timestamps.

**Validates: Requirements 3.9**

### Property 8: Project Data Persistence

*For any* project, all associated data (chats, artifacts, notes, hypotheses) should be persisted and retrievable.

**Validates: Requirements 4.9**

### Property 9: Project Separation

*For any* two different projects, their chat histories and artifact collections should remain completely separate.

**Validates: Requirements 4.5, 4.6**

### Property 10: AI Output Edit Preservation

*For any* AI output that is edited by a user, the system should preserve both the original version and the edited version with timestamps and editor information.

**Validates: Requirements 5.2, 5.8**

### Property 11: Edit Reversion

*For any* edited AI output, the user should be able to revert to the original AI-generated content.

**Validates: Requirements 5.9**

### Property 12: Evidence Citation Completeness

*For any* AI-generated conclusion, the system should provide evidence citations from specific data artifacts or tool results.

**Validates: Requirements 6.1, 6.7**

### Property 13: Uncertainty Indication

*For any* conclusion with confidence below 70%, the system should explicitly mark it as uncertain.

**Validates: Requirements 6.3**

### Property 14: Conflict Identification

*For any* project where data sources contain conflicting information, the system should identify and present both perspectives.

**Validates: Requirements 6.5**

### Property 15: Conclusion Type Classification

*For any* AI-generated conclusion, it should be classified as either an established fact, inference, or speculative reasoning.

**Validates: Requirements 6.6**

### Property 16: Hypothesis Status Tracking

*For any* hypothesis, the system should maintain its current status and timestamp all status changes.

**Validates: Requirements 7.3, 7.6**

### Property 17: Hypothesis Linking

*For any* hypothesis, it should be correctly linked to associated chat messages, notes, or data.

**Validates: Requirements 7.4**

### Property 18: Multi-Source Synthesis

*For any* project containing multiple data artifacts, the AI reasoning engine should reference all relevant sources in its explanations.

**Validates: Requirements 8.6, 10.5**

### Property 19: Protein Behavior Analysis Completeness

*For any* protein stability analysis where temperature, pH, or cellular crowding data is present, the system should consider and reference these factors.

**Validates: Requirements 8.4**

### Property 20: Disease Protein Analysis

*For any* disease-related protein analysis, when abnormal conformations or aggregation patterns are present in the data, the system should identify and report them.

**Validates: Requirements 8.8**

### Property 21: 3D Structure Annotation Persistence

*For any* 3D structure annotation created by a user, it should be saved with the structure and retrievable when the structure is viewed again.

**Validates: Requirements 9.5, 9.6**

### Property 22: Structure Image Export

*For any* annotated 3D structure, the user should be able to export it as a high-resolution image with all annotations visible.

**Validates: Requirements 9.7**

### Property 23: Complementary Information Detection

*For any* project with multiple data sources, the system should identify and flag complementary information from different sources.

**Validates: Requirements 10.3**

### Property 24: Contradictory Information Detection

*For any* project with multiple data sources, the system should identify and flag contradictory information from different sources.

**Validates: Requirements 10.4**

## Error Handling

### Error Categories

1. **Validation Errors**: Invalid file formats, oversized files, malformed data
2. **Connection Errors**: External tool connection failures, timeouts, authentication failures
3. **Processing Errors**: AI reasoning failures, parsing errors, insufficient data
4. **Storage Errors**: Database failures, file storage failures, quota exceeded
5. **Authorization Errors**: Unauthorized access attempts, expired sessions

### Error Handling Strategy

**Validation Errors**:
- Validate all inputs at the API boundary
- Return descriptive error messages to users
- Log validation failures for monitoring
- Reject invalid requests before processing

**Connection Errors**:
- Implement retry logic with exponential backoff (3 attempts)
- Set timeouts for external tool calls (30 seconds default)
- Cache tool status to avoid repeated failed connections
- Provide fallback behavior when tools are unavailable
- Log all connection failures with context

**Processing Errors**:
- Wrap AI reasoning calls in try-catch blocks
- Implement graceful degradation (partial results when possible)
- Log processing errors with full context
- Notify users when analysis cannot be completed
- Provide suggestions for resolution (e.g., "add more data")

**Storage Errors**:
- Implement transaction rollback for database operations
- Retry transient storage failures (3 attempts)
- Monitor storage quotas and warn users proactively
- Maintain data consistency through atomic operations
- Log all storage errors with operation details

**Authorization Errors**:
- Validate user permissions before all operations
- Return 401/403 status codes appropriately
- Log unauthorized access attempts for security monitoring
- Implement session timeout and refresh mechanisms

### Error Response Format

All errors should follow a consistent format:

```typescript
interface ErrorResponse {
  error: {
    code: string
    message: string
    details?: any
    timestamp: Date
    requestId: string
  }
}
```

## Testing Strategy

### Dual Testing Approach

The system will employ both unit testing and property-based testing for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs

These approaches are complementary. Unit tests catch concrete bugs and validate specific scenarios, while property tests verify general correctness across a wide range of inputs.

### Unit Testing

**Focus Areas**:
- Specific examples demonstrating correct behavior
- Edge cases (empty inputs, boundary values, special characters)
- Error conditions (invalid inputs, connection failures, timeouts)
- Integration points between components
- Mock external dependencies (AI services, external tools)

**Example Unit Tests**:
- Upload a valid PDF file and verify it's stored correctly
- Upload a 101MB file and verify it's rejected
- Create a session with empty name and verify validation error
- Request analysis with no artifacts and verify appropriate error
- Test AlphaFold adapter with mock responses

### Property-Based Testing

**Configuration**:
- Use a property-based testing library appropriate for the implementation language
- Configure each test to run minimum 100 iterations
- Tag each test with: **Feature: protein-analysis-hub, Property {number}: {property_text}**

**Focus Areas**:
- Universal properties that hold for all inputs
- Comprehensive input coverage through randomization
- Invariants that must be maintained
- Round-trip properties (save/load, serialize/deserialize)

**Example Property Tests**:
- For any valid artifact, upload then download should return identical content
- For any session with N artifacts, analysis should process all N artifacts
- For any report, all conclusions should have at least one citation
- For any session save/load cycle, all data should be preserved

### Integration Testing

**Focus Areas**:
- End-to-end workflows (upload → analyze → generate report)
- External tool integration (with test instances)
- Database transactions and consistency
- File storage operations
- API endpoint behavior

### Test Data Strategy

**Synthetic Data**:
- Generate random protein sequences for testing
- Create mock tool results with varying structures
- Generate test files of different types and sizes

**Real Data Samples**:
- Maintain a small set of real protein data for validation
- Include known edge cases (unusual sequences, large structures)
- Use anonymized research data when possible

**Property Test Generators**:
- Random session configurations (0-100 artifacts)
- Random file types and sizes
- Random tool result structures
- Random conflict scenarios
- Random uncertainty levels

### Test Coverage Goals

- Unit test coverage: >80% of code paths
- Property test coverage: All 22 correctness properties
- Integration test coverage: All major user workflows
- Error path coverage: All error categories and handling paths


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: File Upload and Validation

*For any* file upload attempt, the system should validate both the file format and size, accepting valid files under 100MB and rejecting others with appropriate error messages.

**Validates: Requirements 1.8, 1.9**

### Property 2: Chat Response Timeliness

*For any* user message sent in the chat interface, the AI reasoning engine should respond within 30 seconds.

**Validates: Requirements 1.11**

### Property 3: Chat History Persistence

*For any* project, all chat messages should be persisted and retrievable when the project is reopened.

**Validates: Requirements 1.12**

### Property 4: Tool Result Integration

*For any* external tool result received, it should be imported into the project and made available for AI reasoning.

**Validates: Requirements 2.5**

### Property 5: Tool Connection Error Handling

*For any* external tool connection failure, the system should both log the error with details and send a descriptive notification to the user.

**Validates: Requirements 2.8**

### Property 6: Research Note Linking

*For any* research note created, it should be correctly associated with the current project and any linked chat messages, artifacts, or findings.

**Validates: Requirements 3.3, 3.4, 3.5, 3.6**

### Property 7: Research Note Timestamps

*For any* research note, it should have accurate creation and modification timestamps.

**Validates: Requirements 3.9**

### Property 8: Project Data Persistence

*For any* project, all associated data (chats, artifacts, notes, hypotheses) should be persisted and retrievable.

**Validates: Requirements 4.9**

### Property 9: Project Separation

*For any* two different projects, their chat histories and artifact collections should remain completely separate.

**Validates: Requirements 4.5, 4.6**

### Property 10: AI Output Edit Preservation

*For any* AI output that is edited by a user, the system should preserve both the original version and the edited version with timestamps and editor information.

**Validates: Requirements 5.2, 5.8**

### Property 11: Edit Reversion

*For any* edited AI output, the user should be able to revert to the original AI-generated content.

**Validates: Requirements 5.9**

### Property 12: Evidence Citation Completeness

*For any* AI-generated conclusion, the system should provide evidence citations from specific data artifacts or tool results.

**Validates: Requirements 6.1, 6.7**

### Property 13: Uncertainty Indication

*For any* conclusion with confidence below 70%, the system should explicitly mark it as uncertain.

**Validates: Requirements 6.3**

### Property 14: Conflict Identification

*For any* project where data sources contain conflicting information, the system should identify and present both perspectives.

**Validates: Requirements 6.5**

### Property 15: Conclusion Type Classification

*For any* AI-generated conclusion, it should be classified as either an established fact, inference, or speculative reasoning.

**Validates: Requirements 6.6**

### Property 16: Hypothesis Status Tracking

*For any* hypothesis, the system should maintain its current status and timestamp all status changes.

**Validates: Requirements 7.3, 7.6**

### Property 17: Hypothesis Linking

*For any* hypothesis, it should be correctly linked to associated chat messages, notes, or data.

**Validates: Requirements 7.4**

### Property 18: Multi-Source Synthesis

*For any* project containing multiple data artifacts, the AI reasoning engine should reference all relevant sources in its explanations.

**Validates: Requirements 8.6, 10.5**

### Property 19: Protein Behavior Analysis Completeness

*For any* protein stability analysis where temperature, pH, or cellular crowding data is present, the system should consider and reference these factors.

**Validates: Requirements 8.4**

### Property 20: Disease Protein Analysis

*For any* disease-related protein analysis, when abnormal conformations or aggregation patterns are present in the data, the system should identify and report them.

**Validates: Requirements 8.8**

### Property 21: 3D Structure Annotation Persistence

*For any* 3D structure annotation created by a user, it should be saved with the structure and retrievable when the structure is viewed again.

**Validates: Requirements 9.5, 9.6**

### Property 22: Structure Image Export

*For any* annotated 3D structure, the user should be able to export it as a high-resolution image with all annotations visible.

**Validates: Requirements 9.7**

### Property 23: Complementary Information Detection

*For any* project with multiple data sources, the system should identify and flag complementary information from different sources.

**Validates: Requirements 10.3**

### Property 24: Contradictory Information Detection

*For any* project with multiple data sources, the system should identify and flag contradictory information from different sources.

**Validates: Requirements 10.4**

## Testing Strategy

### Dual Testing Approach

The system will employ both unit testing and property-based testing for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs

These approaches are complementary. Unit tests catch concrete bugs and validate specific scenarios, while property tests verify general correctness across a wide range of inputs.

### Unit Testing

**Framework**: pytest for Python backend, Jest/React Testing Library for frontend

**Focus Areas**:
- Specific examples demonstrating correct behavior
- Edge cases (empty inputs, boundary values, special characters)
- Error conditions (invalid inputs, connection failures, timeouts)
- Integration points between components
- Mock external dependencies (AI services, external tools)

**Example Unit Tests**:
- Upload a valid PDB file and verify it's stored correctly
- Upload a 101MB file and verify it's rejected
- Create a project with empty name and verify validation error
- Request chat response with no artifacts and verify it works
- Test AlphaFold adapter with mock responses
- Edit AI output and verify original is preserved
- Create research note and verify linking works

### Property-Based Testing

**Framework**: Hypothesis (Python) for backend

**Configuration**:
- Use Hypothesis for Python property tests
- Configure each test to run minimum 100 iterations
- Tag each test with: **Feature: protein-analysis-hub, Property {number}: {property_text}**

**Focus Areas**:
- Universal properties that hold for all inputs
- Comprehensive input coverage through randomization
- Invariants that must be maintained
- Round-trip properties (save/load, serialize/deserialize)

**Example Property Tests**:
- For any valid artifact, upload then download should return identical content
- For any project with N artifacts, all N should be retrievable
- For any chat message edit, original should be preserved and revertible
- For any project save/load cycle, all data should be preserved
- For any research note with links, all links should be bidirectional
- For any hypothesis status change, timestamp should be updated

### Integration Testing

**Focus Areas**:
- End-to-end workflows (upload → chat → analyze → export)
- External tool integration (with test instances or mocks)
- Database transactions and consistency
- File storage operations
- API endpoint behavior
- Frontend-backend integration

### Test Data Strategy

**Synthetic Data**:
- Generate random protein sequences for testing
- Create mock tool results with varying structures
- Generate test files of different types and sizes
- Use Hypothesis strategies for property tests

**Real Data Samples**:
- Maintain a small set of real protein data for validation
- Include known edge cases (unusual sequences, large structures)
- Use anonymized research data when possible

**Property Test Generators**:
```python
from hypothesis import strategies as st

# Project generators
projects
