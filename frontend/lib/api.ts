import { Project, ProjectWithRuns, CreateProjectRequest, UpdateProjectRequest } from './types'
import { config } from '@/app/config'

interface Metadata {
  [key: string]: unknown;
}

interface SessionResponse {
  session_id: string;
  jwt_token: string;
  refresh_token: string;
  expires_in: number;
  status: string;
}

interface RefreshResponse {
  jwt_token: string;
  refresh_token: string;
  expires_in: number;
}

export interface RunStatusResponse {
  run_id: string;
  exists: boolean;
  state?: string;
  can_reconnect: boolean;
  connected_at?: string;
  disconnected_at?: string;
  grace_period_expires?: string;
  buffered_events?: number;
  last_checkpoint?: string;
  message?: string;
}

export class ProjectService {
  // Generic fetch method with credentials
  private static async fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${config.api.baseUrl}${endpoint}`

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      credentials: 'include', // Include cookies for fingerprint
      ...options,
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }

  // Get all projects
  static async getAllProjects(): Promise<ProjectWithRuns[]> {
    return this.fetchApi<ProjectWithRuns[]>('/projects')
  }

  // Get single project details
  static async getProject(projectId: string): Promise<ProjectWithRuns> {
    return this.fetchApi<ProjectWithRuns>(`/project/${projectId}`)
  }

  // Create new project
  static async createProject(data: CreateProjectRequest): Promise<{ message: string; data: Project }> {
    console.log('[DEBUG] ProjectService.createProject called with:', data)
    const result = await this.fetchApi<{ message:string; data: Project }>('/project', {
      method: 'POST',
      body: JSON.stringify(data),
    })
    console.log('[DEBUG] createProject result:', result)
    return result
  }

  // Update project
  static async updateProject(projectId: string, data: UpdateProjectRequest): Promise<{ message: string }> {
    return this.fetchApi<{ message: string }>(`/project/${projectId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
  }

  // Delete project
  static async deleteProject(projectId: string): Promise<{ message: string }> {
    return this.fetchApi<{ message: string }>(`/project/${projectId}`, {
      method: 'DELETE',
    })
  }

  // Get metadata
  static async getMetadata(params?: Record<string, string>): Promise<Metadata> {
    const query = params ? `?${new URLSearchParams(params).toString()}` : ''
    return this.fetchApi<Metadata>(`/metadata${query}`)
  }

  // Create session (with JWT tokens)
  static async createSession(projectId: string = 'default'): Promise<SessionResponse> {
    return this.fetchApi<SessionResponse>('/session', {
      method: 'POST',
      body: JSON.stringify({ project_id: projectId }),
    })
  }

  // Refresh session tokens
  static async refreshSession(refreshToken: string): Promise<RefreshResponse> {
    return this.fetchApi<RefreshResponse>('/session/refresh', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    })
  }

  // Get run connection status
  static async getRunStatus(runId: string): Promise<RunStatusResponse> {
    return this.fetchApi<RunStatusResponse>(`/run/${runId}/status`)
  }

  // Update run metadata
  static async updateRun(runId: string, metadata: Record<string, unknown>): Promise<{ message: string }> {
    return this.fetchApi<{ message: string }>(`/run/${runId}`, {
      method: 'PUT',
      body: JSON.stringify(metadata),
    })
  }

  // Rename run
  static async renameRun(runId: string, newName: string): Promise<{
    message: string;
    old_filename: string;
    new_filename: string
  }> {
    return this.fetchApi<{
      message: string;
      old_filename: string;
      new_filename: string
    }>(`/run/${runId}/name`, {
      method: 'PUT',
      body: JSON.stringify({ new_name: newName }),
    })
  }

  // Delete run
  static async deleteRun(runId: string): Promise<{ message: string }> {
    return this.fetchApi<{ message: string }>(`/run/${runId}`, {
      method: 'DELETE',
    })
  }

  // Move run to another project
  static async moveRun(runId: string, fromProjectId: string, toProjectId: string): Promise<{
    message: string;
    old_filename: string;
    new_filename: string;
    source_project: string;
    destination_project: string
  }> {
    return this.fetchApi<{
      message: string;
      old_filename: string;
      new_filename: string;
      source_project: string;
      destination_project: string
    }>('/run/move', {
      method: 'POST',
      body: JSON.stringify({
        run_id: runId,
        from_project_id: fromProjectId,
        to_project_id: toProjectId
      }),
    })
  }
}