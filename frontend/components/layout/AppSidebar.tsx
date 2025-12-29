"use client"

import { ChevronDown, Plus, SquarePen, Trash2, MoreHorizontal, Siren, Bot, Frame, Cog, ExternalLink, Twitter, Rocket } from "lucide-react"
import Image from "next/image"
import React, { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { observer } from 'mobx-react-lite'
import { projectStore } from "@/app/stores/projectStore"
import { selectionStore } from "@/app/stores/selectionStore"
import { removeIicExtension } from "@/lib/utils"

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuAction,
  SidebarMenuSubItem,
  SidebarMenuSubButton,
} from "@/components/ui/sidebar"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Collapsible,
  CollapsibleContent,
} from "@/components/ui/collapsible"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

// Type definitions
interface ModelProvider {
  id: string
  name: string
  enabled: boolean
  apiKey: string
  description: string
  url: string
}

interface GeneralSetting {
  id: string
  label: string
  description: string
  type: 'switch' | 'select'
  value: boolean | string
  options?: string[]
}

interface LinkItem {
  id: string
  label: string
  url: string
  icon: string
}

interface AboutInfo {
  appInfo: {
    name: string
    version: string
    buildDate: string
    icon: string
  }
  links: LinkItem[]
  community: {
    description: string
    discordUrl: string
  }
}

interface SettingsData {
  modelProviders: Record<string, ModelProvider>
  generalSettings: Record<string, GeneralSetting>
  aboutInfo: AboutInfo
}

export const AppSidebar = observer(function AppSidebar() {
  const [isNewProjectOpen, setIsNewProjectOpen] = useState(false)
  const [isEditProjectOpen, setIsEditProjectOpen] = useState(false)
  const [isEditFileOpen, setIsEditFileOpen] = useState(false)
  const [editingProject, setEditingProject] = useState<{ id: string; name: string } | null>(null)
  const [editingFile, setEditingFile] = useState<{ runId: string; filename: string } | null>(null)
  const [newProjectName, setNewProjectName] = useState("")
  const [editProjectName, setEditProjectName] = useState("")
  const [editFileName, setEditFileName] = useState("")
  const [isDeleteProjectConfirmOpen, setIsDeleteProjectConfirmOpen] = useState(false)
  const [projectToDeleteId, setProjectToDeleteId] = useState<string | null>(null)
  const [isDeleteFileConfirmOpen, setIsDeleteFileConfirmOpen] = useState(false)
  const [fileToDeleteId, setFileToDeleteId] = useState<string | null>(null)
  const [draggedFile, setDraggedFile] = useState<{
    runId: string;
    filename: string;
    fromProjectId: string;
  } | null>(null)
  const [dragOverProject, setDragOverProject] = useState<string | null>(null)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [activeSettingTab, setActiveSettingTab] = useState('model-provider')
  const [modelProviders, setModelProviders] = useState<Record<string, ModelProvider>>({})
  const [generalSettings, setGeneralSettings] = useState<Record<string, GeneralSetting>>({})
  const [aboutInfo, setAboutInfo] = useState<AboutInfo | null>(null)
  const [expandedProvider, setExpandedProvider] = useState<string | null>(null)
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({})
  const [settingsLoading, setSettingsLoading] = useState(true)
  const router = useRouter()

  // Directly get data from the store, methods are called via projectStore.methodName()
  const { projects, loading } = projectStore

  // Load settings data
  const loadSettings = async () => {
    try {
      setSettingsLoading(true)
      // Simulate loading data from an API, actually loads from a mock file
      const response = await fetch('/mock/settings.json')
      const data: SettingsData = await response.json()
      setModelProviders(data.modelProviders)
      setGeneralSettings(data.generalSettings || {})
      setAboutInfo(data.aboutInfo || null)

      // Initialize API Keys
      const keys: Record<string, string> = {}
      Object.values(data.modelProviders).forEach((provider: ModelProvider) => {
        keys[provider.id] = provider.apiKey || ''
      })
      setApiKeys(keys)
    } catch (error) {
      console.error('Failed to load settings:', error)
    } finally {
      setSettingsLoading(false)
    }
  }

  // Model Provider handler functions
  const toggleProvider = (providerId: string) => {
    setModelProviders((prev: Record<string, ModelProvider>) => ({
      ...prev,
      [providerId]: {
        ...prev[providerId],
        enabled: !prev[providerId].enabled
      }
    }))
  }

  const toggleProviderSettings = (providerId: string) => {
    setExpandedProvider(prev => prev === providerId ? null : providerId)
  }

  const updateApiKey = (providerId: string, value: string) => {
    setApiKeys(prev => ({ ...prev, [providerId]: value }))
  }

  const saveApiKey = (providerId: string) => {
    // API call to save can be made here
    console.log(`Saving API key for ${providerId}:`, apiKeys[providerId])
    // Simulate successful save
    setModelProviders((prev: Record<string, ModelProvider>) => ({
      ...prev,
      [providerId]: {
        ...prev[providerId],
        apiKey: apiKeys[providerId]
      }
    }))
  }

  // General Settings handler functions
  const updateGeneralSetting = (settingId: string, value: boolean | string) => {
    setGeneralSettings((prev: Record<string, GeneralSetting>) => ({
      ...prev,
      [settingId]: {
        ...prev[settingId],
        value
      }
    }))

    // API call to save settings can be made here
    console.log(`Updating setting ${settingId} to:`, value)
  }

  // About handler functions
  const handleCheckUpdate = () => {
    console.log('Checking for updates...')
    // Logic for checking updates can be implemented here
  }

  const handleLinkClick = (url: string) => {
    window.open(url, '_blank')
  }

  const handleJoinDiscord = () => {
    if (aboutInfo?.community.discordUrl) {
      window.open(aboutInfo.community.discordUrl, '_blank')
    }
  }

  // Return the corresponding icon component based on the icon string
  const getIconComponent = (iconName: string) => {
    switch (iconName) {
      case 'twitter':
        return <Twitter className="h-4 w-4" />
      case 'rocket':
        return <Rocket className="h-4 w-4" />
      default:
        return <ExternalLink className="h-4 w-4" />
    }
  }

  // Load data when the Settings Dialog is opened
  React.useEffect(() => {
    if (isSettingsOpen && settingsLoading) {
      loadSettings()
    }
  }, [isSettingsOpen, settingsLoading])

  // Settings options configuration
  const settingsTabs = [
    { id: 'model-provider', label: 'Model Provider', icon: Bot },
    { id: 'general-settings', label: 'General Settings', icon: Frame },
    // { id: 'about', label: 'About', icon: Info }
  ]

  const toggleProject = (projectId: string) => {
    selectionStore.toggleProject(projectId)
  }

  // Create new project
  const handleCreateProject = async () => {
    console.log('[DEBUG] handleCreateProject called, newProjectName:', JSON.stringify(newProjectName))
    if (!newProjectName.trim()) {
      console.log('[DEBUG] newProjectName is empty, returning')
      return
    }

    try {
      console.log('[DEBUG] Calling projectStore.createProject with:', { name: newProjectName.trim() })
      await projectStore.createProject({ name: newProjectName.trim() })
      setNewProjectName("")
      setIsNewProjectOpen(false)
    } catch (error) {
      console.error('Failed to create project:', error)
      // Error hints can be added here
    }
  }

  // Edit project
  const handleEditProject = (project: { project_id: string; name: string }) => {
    // Get the currently displayed project name - prioritize the latest name from selectionStore
    const getCurrentProjectName = (projectId: string, dbName: string) => {
      if (selectionStore.selectedProject?.projectId === projectId) {
        return selectionStore.selectedProject.projectName;
      }
      if (selectionStore.selectedFile?.projectId === projectId) {
        return selectionStore.selectedFile.projectName;
      }
      return dbName;
    };

    const currentProjectName = getCurrentProjectName(project.project_id, project.name);

    // Use setTimeout to ensure the DropdownMenu is fully closed before opening the Dialog
    setTimeout(() => {
      setEditingProject({ id: project.project_id, name: currentProjectName })
      setEditProjectName(currentProjectName)
      setIsEditProjectOpen(true)
    }, 0)
  }

  const handleUpdateProject = async () => {
    if (!editingProject || !editProjectName.trim()) return

    try {
      console.log('AppSidebar updating:', editProjectName.trim());

      // Only call the API, subsequent synchronization relies entirely on projectStore.updateProject() → loadProjects() → updateProjectsMap()
      await projectStore.updateProject(editingProject.id, { name: editProjectName.trim() })

      setEditingProject(null)
      setEditProjectName("")
      setIsEditProjectOpen(false)
    } catch (error) {
      console.error('Failed to update project:', error)
      // Error hints can be added here
    }
  }

  // Delete project
  const handleConfirmDeleteProject = async () => {
    if (!projectToDeleteId) return

    try {
      await projectStore.deleteProject(projectToDeleteId)

      // After successful deletion, check and clear related selection states
      const deletedProjectId = projectToDeleteId;

      // If the currently selected project is the one being deleted, clear the selection state
      if (selectionStore.selectedProject?.projectId === deletedProjectId) {
        selectionStore.clearSelection();
      }

      // If the currently selected file belongs to the deleted project, also clear the selection state
      if (selectionStore.selectedFile?.projectId === deletedProjectId) {
        selectionStore.clearSelection();
      }

    } catch (error) {
      console.error('Failed to delete project:', error)
      // Error hints can be added here
    } finally {
      setIsDeleteProjectConfirmOpen(false)
      setProjectToDeleteId(null)
    }
  }

  const triggerDeleteProject = (projectId: string) => {
    setTimeout(() => {
      setProjectToDeleteId(projectId)
      setIsDeleteProjectConfirmOpen(true)
    }, 0)
  }



  // Handle file selection - use router navigation
  const handleFileSelect = (run: { filename?: string; meta: { run_id?: string; description?: string } }, projectData: { project: { project_id: string; name: string } }, index: number) => {
    const runId = run.meta.run_id || `${projectData.project.project_id}-${index}`;

    if (runId) {
      // Navigate to the run page
              router.push(`/r?id=${runId}`);
    }
  };

  // Handle project selection - add a new handler function
  const handleProjectSelect = (project: { project_id: string; name: string }) => {
    // Navigate to the project page
            router.push(`/p?id=${project.project_id}`);
  };

  // Edit file name
  const handleEditFile = (run: { filename?: string; meta: { run_id?: string; description?: string } }) => {
    const filename = run.filename || run.meta.description || 'Untitled';
    const runId = run.meta.run_id;
    if (runId) {
      setEditingFile({ runId, filename })
      setEditFileName(removeIicExtension(filename)) // Remove file extension
      setIsEditFileOpen(true)
    }
  }

  const handleUpdateFileName = async () => {
    if (!editingFile || !editFileName.trim()) return

    try {
      await projectStore.renameRun(editingFile.runId, editFileName.trim())
      setEditingFile(null)
      setEditFileName("")
      setIsEditFileOpen(false)
    } catch (error) {
      console.error('Failed to rename file:', error)
      // Error hints can be added here
    }
  }

  // Delete file
  const handleConfirmDeleteFile = async () => {
    if (!fileToDeleteId) return

    try {
      await projectStore.deleteRun(fileToDeleteId)
    } catch (error) {
      console.error('Failed to delete file:', error)
      // Error hints can be added here
    } finally {
      setIsDeleteFileConfirmOpen(false)
      setFileToDeleteId(null)
    }
  }

  const triggerDeleteFile = (runId: string | undefined) => {
    if (runId) {
      setTimeout(() => {
        setFileToDeleteId(runId)
        setIsDeleteFileConfirmOpen(true)
      }, 0)
    } else {
      console.error("Delete failed: run_id is missing.")
      // User prompts can be added here, e.g., using toast
    }
  }

  // Drag start
  const handleDragStart = (e: React.DragEvent, run: { filename?: string; meta: { run_id?: string; description?: string } }, projectId: string) => {
    const runId = run.meta.run_id
    const filename = run.filename || run.meta.description || 'Untitled'

    // If there is no run_id, it cannot be dragged
    if (!runId) {
      e.preventDefault()
      return
    }

    setDraggedFile({
      runId,
      filename,
      fromProjectId: projectId
    })

    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', runId)
  }

  // Drag end
  const handleDragEnd = () => {
    setDraggedFile(null)
    setDragOverProject(null)
  }

  // Allow drop
  const handleDragOver = (e: React.DragEvent, projectId: string) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'

    // Cannot drag to the same project
    if (draggedFile && draggedFile.fromProjectId !== projectId) {
      setDragOverProject(projectId)
    }
  }

  // Leave drag area
  const handleDragLeave = () => {
    setDragOverProject(null)
  }

  // Handle drop
  const handleDrop = async (e: React.DragEvent, toProjectId: string) => {
    e.preventDefault()
    setDragOverProject(null)

    if (!draggedFile || draggedFile.fromProjectId === toProjectId) {
      return
    }

    try {
      await projectStore.moveRun(draggedFile.runId, draggedFile.fromProjectId, toProjectId)
      // Clear selection state on success
      if (selectionStore.selectedFile?.runId === draggedFile.runId) {
        selectionStore.clearSelection()
      }
    } catch (error) {
      console.error('Failed to move file:', error)
      // Can add error toast notification
    } finally {
      setDraggedFile(null)
    }
  }

  return (
    <Sidebar>
      <SidebarHeader className="py-3 px-4 h-12">
        <Link href="/" className="flex items-center gap-3">
          <Image src="/webview/logo.png" alt="Common Ground" width={28} height={28} className="rounded" />
          <span className="font-medium text-lg">Common Ground</span>
        </Link>
      </SidebarHeader>

      <SidebarContent>
        {/* New Project Button */}
        <div className="px-3 pt-2">
          <Dialog open={isNewProjectOpen} onOpenChange={setIsNewProjectOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" className="w-full justify-start gap-2">
                <Plus className="h-4 w-4" />
                New Project
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>New Project</DialogTitle>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <div className="flex gap-1 items-center col-span-4">
                    <div className="flex items-center justify-center h-9 w-9 rounded-md border border-input bg-background">
                      <div className="text-lg">🤖</div>
                    </div>
                    <Input
                      id="name"
                      placeholder="Project name"
                      className="flex-1"
                      value={newProjectName}
                      onChange={(e) => setNewProjectName(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleCreateProject()}
                    />
                  </div>
                </div>
                <div className="flex gap-3 items-start rounded-lg bg-[#F9F9F9] p-3">
                  <div className="mt-1">
                    <Siren size={22} strokeWidth={1.2} />
                  </div>
                  <div>
                    <div className="text-sm font-medium">What are Projects?</div>
                    <div className="text-sm text-muted-foreground">
                      Projects let you save chats, files, and custom instructions together for ongoing work, or just to keep things organized.
                    </div>
                  </div>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsNewProjectOpen(false)}>
                  Cancel
                </Button>
                <Button type="submit" onClick={handleCreateProject}>
                  Create
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>

        {/* Edit Project Dialog */}
        <Dialog open={isEditProjectOpen} onOpenChange={setIsEditProjectOpen}>
          <DialogTrigger asChild>
            <div style={{ display: 'none' }} />
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit Project</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <div className="flex gap-1 items-center col-span-4">
                  <div className="flex items-center justify-center h-9 w-9 rounded-md border border-input bg-background">
                    <div className="text-lg">🤖</div>
                  </div>
                  <Input
                    id="edit-name"
                    placeholder="Project name"
                    className="flex-1"
                    value={editProjectName}
                    onChange={(e) => setEditProjectName(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleUpdateProject()}
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsEditProjectOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" onClick={handleUpdateProject}>
                Save
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Project Confirmation Dialog */}
        <Dialog open={isDeleteProjectConfirmOpen} onOpenChange={setIsDeleteProjectConfirmOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete Project</DialogTitle>
            </DialogHeader>
            <div className="py-4">
              <p>Are you sure you want to delete this project? This will delete all associated files and cannot be undone.</p>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsDeleteProjectConfirmOpen(false)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleConfirmDeleteProject}>
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>



        {/* Edit File Dialog */}
        <Dialog open={isEditFileOpen} onOpenChange={setIsEditFileOpen}>
          <DialogTrigger asChild>
            <div style={{ display: 'none' }} />
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit File Name</DialogTitle>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <div className="flex gap-1 items-center col-span-4">
                  <div className="flex items-center justify-center h-9 w-9 rounded-md border border-input bg-background">
                    <div className="text-lg">📄</div>
                  </div>
                  <Input
                    id="edit-file-name"
                    placeholder="File name"
                    className="flex-1"
                    value={editFileName}
                    onChange={(e) => setEditFileName(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleUpdateFileName()}
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsEditFileOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" onClick={handleUpdateFileName}>
                Save
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete File Confirmation Dialog */}
        <Dialog open={isDeleteFileConfirmOpen} onOpenChange={setIsDeleteFileConfirmOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Delete File</DialogTitle>
            </DialogHeader>
            <div className="py-4">
              <p>Are you sure you want to delete this file? This action cannot be undone.</p>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsDeleteFileConfirmOpen(false)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleConfirmDeleteFile}>
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Projects */}
        <SidebarGroup>
          <SidebarGroupLabel>Project</SidebarGroupLabel>
          <SidebarGroupContent>
            {loading ? (
              <div className="px-2 py-4 text-sm text-muted-foreground">Loading projects...</div>
            ) : (
              <SidebarMenu>
                {projects
                  .filter((projectData) => projectData.project.deleted !== "true")
                  .slice().sort((a, b) => {
                    // Default Project always comes first
                    if (a.project.project_id === 'default') return -1;
                    if (b.project.project_id === 'default') return 1;

                    // Other projects are sorted by created_at in descending order (newest first)
                    const dateA = new Date(a.project.created_at);
                    const dateB = new Date(b.project.created_at);
                    return dateB.getTime() - dateA.getTime();
                  })
                  .map((projectData) => {
                    // Get project display name - prioritize using the name from selectionStore for real-time synchronization
                    const getProjectDisplayName = (projectId: string, dbName: string) => {
                      if (selectionStore.selectedProject?.projectId === projectId) {
                        return selectionStore.selectedProject.projectName;
                      }
                      if (selectionStore.selectedFile?.projectId === projectId) {
                        return selectionStore.selectedFile.projectName;
                      }
                      return dbName;
                    };

                    const displayProjectName = getProjectDisplayName(
                      projectData.project.project_id,
                      projectData.project.name
                    );

                    return (
                      <Collapsible
                        open={selectionStore.isProjectOpen(projectData.project.project_id)}
                        className="w-full group/collapsible"
                        key={projectData.project.project_id}
                      >
                        <SidebarMenuItem>
                          <SidebarMenuButton
                            onClick={() => handleProjectSelect(projectData.project)}
                            className={`w-full relative hover:bg-accent transition-all duration-200 ${
                              dragOverProject === projectData.project.project_id ? 'bg-blue-100 border-2 border-blue-300 border-dashed scale-[1.02]' : ''
                            }`}
                            onDragOver={(e) => handleDragOver(e, projectData.project.project_id)}
                            onDragLeave={handleDragLeave}
                            onDrop={(e) => handleDrop(e, projectData.project.project_id)}
                          >
                            <div className="flex items-center flex-1">
                              <div
                                onClick={(e) => {
                                  e.stopPropagation()
                                  toggleProject(projectData.project.project_id)
                                }}
                                className="p-1 cursor-pointer hover:bg-neutral-200 hover:rounded"
                              >
                                <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${selectionStore.isProjectOpen(projectData.project.project_id) ? "" : "-rotate-90"}`} />
                              </div>
                              <span className="ml-1">{displayProjectName}</span>
                            </div>
                          </SidebarMenuButton>
                          {/* Do not show edit and delete menu for Default Project */}
                          {projectData.project.project_id !== 'default' && (
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <SidebarMenuAction className="right-8">
                                  <MoreHorizontal className="h-4 w-4 opacity-0 group-hover/collapsible:opacity-100" />
                                </SidebarMenuAction>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent side="right" align="start">
                                <DropdownMenuItem onClick={() => handleEditProject(projectData.project)}>
                                  <SquarePen className="mr-2 h-4 w-4" />
                                  <span>Edit Project</span>
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  className="text-red-600 focus:text-red-600 focus:bg-red-50"
                                  onClick={() => triggerDeleteProject(projectData.project.project_id)}
                                >
                                  <Trash2 className="mr-2 h-4 w-4" />
                                  <span>Delete Project</span>
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          )}
                          <SidebarMenuAction onClick={() => router.push(`/p?id=${projectData.project.project_id}`)}>
                            <Plus className="h-4 w-4 opacity-0 group-hover/collapsible:opacity-100" />
                          </SidebarMenuAction>
                          <CollapsibleContent>
                            <SidebarMenuSub className="mx-0 pr-0">
                              {projectData.runs
                                .slice()
                                // Deduplicate runs by run_id (keep first occurrence)
                                .filter((run, index, self) => {
                                  const runId = run.meta?.run_id;
                                  if (!runId) return true; // Keep runs without run_id
                                  return self.findIndex(r => r.meta?.run_id === runId) === index;
                                })
                                .sort((a, b) => {
                                  // Files are sorted by created in descending order (newest first)
                                  const getCreated = (run: { meta?: { created?: string; [key: string]: unknown } }) => {
                                    const meta = run.meta || {};
                                    return meta.created || null;
                                  };

                                  const timeA = getCreated(a);
                                  const timeB = getCreated(b);

                                  // If both have created, sort by time in descending order
                                  if (timeA && timeB) {
                                    return new Date(timeB).getTime() - new Date(timeA).getTime();
                                  }

                                  // Those with created come first
                                  if (timeA && !timeB) return -1;
                                  if (!timeA && timeB) return 1;

                                  // If neither has created, maintain original order
                                  return 0;
                                })
                                .map((run, index) => {
                                const isSelected = selectionStore.selectedFile?.runId === (run.meta.run_id || `${projectData.project.project_id}-${index}`);
                                // Prioritize using meta.display_name as the display name.
                                // If display_name does not exist, fall back to using filename.
                                // If filename also does not exist, use description or 'Untitled' as a final fallback.
                                const displayNameFromMeta = typeof run.meta?.display_name === 'string' ? run.meta.display_name : null;
                                const filename = run.filename;
                                const description = run.meta?.description;

                                const fullName = displayNameFromMeta || filename || description || 'Untitled';

                                // Improved file name display logic, remove .iic extension
                                const displayFileName = removeIicExtension(fullName);

                                return (
                                  <SidebarMenuSubItem key={`${projectData.project.project_id}-${run.meta.run_id || run.filename || index}`} className="group/item">
                                    <div className="relative flex items-center w-full hover:bg-accent/50 rounded-sm transition-colors">
                                      <SidebarMenuSubButton
                                        className={`cursor-pointer pl-6 text-sm relative before:absolute before:left-[7px] before:top-[50%] before:w-3 before:h-px before:bg-border flex-1 min-w-0 text-foreground ${
                                          isSelected ? 'bg-accent' : ''
                                        } ${draggedFile?.runId === run.meta.run_id ? 'opacity-50 transition-opacity duration-200' : ''}`}
                                        title={displayFileName}
                                        onClick={() => handleFileSelect(run, projectData, index)}
                                        draggable={!!run.meta.run_id}
                                        onDragStart={(e) => handleDragStart(e, run, projectData.project.project_id)}
                                        onDragEnd={handleDragEnd}
                                      >
                                        <span className="truncate block">{displayFileName}</span>
                                      </SidebarMenuSubButton>

                                        {/* File action icons */}
                                        <div className="absolute top-1/2 -translate-y-1/2 right-1.5 flex items-center gap-0.5 opacity-0 group-hover/item:opacity-100 transition-opacity p-0.5 rounded-md bg-accent">
                                          <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-6 w-6 rounded-sm hover:bg-neutral-200"
                                            onClick={(e) => {
                                              e.stopPropagation()
                                              handleEditFile(run)
                                            }}
                                            title="Edit file"
                                          >
                                            <SquarePen className="h-4 w-4" />
                                          </Button>
                                          <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-6 w-6 rounded-sm hover:text-red-500 hover:bg-neutral-200"
                                            onClick={(e) => {
                                              e.stopPropagation()
                                              triggerDeleteFile(run.meta.run_id)
                                            }}
                                            title="Delete file"
                                          >
                                            <Trash2 className="h-4 w-4" />
                                          </Button>
                                        </div>
                                      </div>
                                    </SidebarMenuSubItem>
                                  );
                                })}
                            </SidebarMenuSub>
                          </CollapsibleContent>
                        </SidebarMenuItem>
                      </Collapsible>
                    );
                  })}
              </SidebarMenu>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      {/* <div className="flex flex-col mt-auto">*/}
        {/* Marketplace & Tools */}
        {/* <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <Link href="/my-agent" className="w-full">
                  <SidebarMenuButton className={pathname === '/my-agent' ? 'bg-accent' : ''}>
                    <Store className="h-4 w-4" />
                    <span>My Agent</span>
                  </SidebarMenuButton>
                </Link>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton>
                  <Drill className="h-4 w-4" />
                  <span>Tool Set</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton onClick={() => setIsSettingsOpen(true)}>
                  <Settings className="h-4 w-4" />
                  <span>Settings</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>*/}

        {/* Early Adopters */}
        {/* <div className="p-3">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="w-full justify-start gap-2 px-2 focus-visible:ring-0 focus-visible:ring-offset-0 focus:outline-none"
              >
                <Avatar className="h-8 w-8 rounded-md">
                  <AvatarImage src="/webview/logo.png" alt="Early adopters" />
                  <AvatarFallback>EA</AvatarFallback>
                </Avatar>
                <span className="font-medium text-sm">Early adopters</span>
                <MoreVertical className="h-4 w-4 ml-auto" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56" align="start" side="right">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <User className="mr-2 h-4 w-4" />
                <span>Personal Information</span>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="mr-2 h-4 w-4" />
                <span>Settings</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <LogOut className="mr-2 h-4 w-4" />
                <span>Log Out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>*/}

      {/* Settings Dialog */}
      <Dialog open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
        <DialogContent className="w-[1119px] h-[692px] max-w-none p-0">
          <DialogTitle className="sr-only">Settings</DialogTitle>
          <div className="flex h-full">
            {/* Left Sidebar */}
            <div className="w-[255px] bg-gray-50 border-r rounded-l-lg">
              <div className="p-2">
                {settingsTabs.map((tab) => {
                  const Icon = tab.icon
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveSettingTab(tab.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        activeSettingTab === tab.id
                          ? 'bg-gray-100 text-gray-900'
                          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                      {tab.label}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Right Content */}
            <div className="flex-1 p-6">
              {activeSettingTab === 'model-provider' && (
                <div>
                  <h3 className="text-2xl font-semibold mb-2">Model Provider</h3>
                  <p className="text-gray-500 text-sm mb-8">
                    Receive emails about new products, features, and more.
                  </p>

                  {settingsLoading ? (
                    <div className="text-center py-8">
                      <div className="text-gray-500">Loading...</div>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {Object.values(modelProviders).map((provider: ModelProvider) => (
                        <div key={provider.id}>
                          {/* Provider Item */}
                          <div className="py-4 border-b border-gray-200">
                            <div className="flex items-center justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                  <h4 className="text-base font-medium">{provider.name}</h4>
                                  <ExternalLink className="h-4 w-4 text-gray-400" />
                                </div>
                                <p className="text-sm text-gray-500">{provider.description}</p>
                              </div>
                              <div className="flex items-center gap-3">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0"
                                  onClick={() => toggleProviderSettings(provider.id)}
                                >
                                  <Cog className="h-4 w-4 text-gray-500" />
                                </Button>
                                <Switch
                                  checked={provider.enabled}
                                  onCheckedChange={() => toggleProvider(provider.id)}
                                />
                              </div>
                            </div>
                            {/* API Key Configuration */}
                            {expandedProvider === provider.id && (
                              <div className="p-4 bg-gray-50 border border-gray-200 rounded-md mt-2">
                                <div className="mb-3">
                                  <label className="text-sm font-medium text-gray-700">API Key</label>
                                </div>
                                <div className="flex gap-2">
                                  <Input
                                    type="password"
                                    placeholder="Enter your API key"
                                    value={apiKeys[provider.id] || ''}
                                    onChange={(e) => updateApiKey(provider.id, e.target.value)}
                                    className="flex-1"
                                  />
                                  <Button
                                    variant="outline"
                                    onClick={() => saveApiKey(provider.id)}
                                    className="px-6"
                                  >
                                    Check
                                  </Button>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeSettingTab === 'general-settings' && (
                <div>
                  <h3 className="text-2xl font-semibold mb-2">General Settings</h3>
                  <p className="text-gray-500 text-sm mb-8">
                    Receive emails about new products, features, and more.
                  </p>

                  {settingsLoading ? (
                    <div className="text-center py-8">
                      <div className="text-gray-500">Loading...</div>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {Object.values(generalSettings).map((setting: GeneralSetting) => (
                        <div key={setting.id} className="flex items-center justify-between py-4 border-b border-gray-200">
                          <div className="flex-1">
                            <h4 className="text-base font-medium mb-1">{setting.label}</h4>
                            <p className="text-sm text-gray-500">{setting.description}</p>
                          </div>

                          <div className="flex items-center">
                            {setting.type === 'switch' ? (
                              <Switch
                                checked={setting.value as boolean}
                                onCheckedChange={(checked) => updateGeneralSetting(setting.id, checked)}
                              />
                            ) : setting.type === 'select' ? (
                              <Select
                                value={setting.value as string}
                                onValueChange={(value) => updateGeneralSetting(setting.id, value)}
                              >
                                <SelectTrigger className="w-[200px]">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {setting.options?.map((option) => (
                                    <SelectItem key={option} value={option}>
                                      {option}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            ) : null}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeSettingTab === 'about' && (
                <div>
                  <h3 className="text-2xl font-semibold mb-2">About</h3>
                  <p className="text-gray-500 text-sm mb-8">
                    Receive emails about new products, features, and more.
                  </p>

                  {settingsLoading ? (
                    <div className="text-center py-8">
                      <div className="text-gray-500">Loading...</div>
                    </div>
                  ) : aboutInfo ? (
                    <div className="space-y-8">
                      {/* App Info Section */}
                      <div className="flex items-center justify-between py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-12 h-12 rounded-lg bg-black flex items-center justify-center">
                            <Image
                              src={aboutInfo.appInfo.icon}
                              alt={aboutInfo.appInfo.name}
                              width={32}
                              height={32}
                              className="rounded"
                            />
                          </div>
                          <div>
                            <h4 className="text-base font-medium">{aboutInfo.appInfo.name}</h4>
                            <p className="text-sm text-gray-500">
                              {aboutInfo.appInfo.version} ({aboutInfo.appInfo.buildDate})
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="outline"
                          onClick={handleCheckUpdate}
                          className="px-6"
                        >
                          Check Update
                        </Button>
                      </div>

                      {/* Links Section */}
                      <div>
                        <h4 className="text-lg font-semibold mb-4">Links</h4>
                        <div className="space-y-3">
                          {aboutInfo.links.map((link) => (
                            <button
                              key={link.id}
                              onClick={() => handleLinkClick(link.url)}
                              className="flex items-center gap-3 text-sm text-blue-600 hover:text-blue-800 transition-colors"
                            >
                              {getIconComponent(link.icon)}
                              <span>{link.label}</span>
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Community Section */}
                      <div>
                        <h4 className="text-lg font-semibold mb-3">Community</h4>
                        <div className="flex items-center justify-between">
                          <p className="text-sm text-gray-600 flex-1 mr-4">
                            {aboutInfo.community.description}
                          </p>
                          <Button
                            variant="outline"
                            onClick={handleJoinDiscord}
                            className="px-6 flex-shrink-0"
                          >
                            Join Discord
                          </Button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="text-gray-500">No about information available</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </Sidebar>
  )
});
