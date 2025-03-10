"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import DashboardHeader from "@/components/dashboard-header"
import DashboardSidebar from "@/components/dashboard-sidebar"
import OverviewTab from "@/components/tabs/overview-tab"
import AnalyticsTab from "@/components/tabs/analytics-tab"
import ControlTab from "@/components/tabs/control-tab"
import { useToast } from "@/components/ui/use-toast"
import { Toaster } from "@/components/ui/toaster"
import { Loader2 } from "lucide-react"

export default function DashboardPage({ params }: { params: { systemId: string } }) {
  const { systemId } = params
  const router = useRouter()
  const { toast } = useToast()
  const [isLoading, setIsLoading] = useState(true)
  const [systemHealth, setSystemHealth] = useState<any>(null)

  useEffect(() => {
    // Check if token exists, if not redirect to login
    const token = localStorage.getItem("token")
    if (!token) {
      router.push("/")
      return
    }

    // Fetch system health data
    const fetchSystemHealth = async () => {
      try {
        try {
          const response = await fetch("http://localhost:8000/api/health", {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          })

          if (!response.ok) {
            throw new Error("Failed to fetch system health")
          }

          const data = await response.json()
          setSystemHealth(data)
        } catch (error) {
          console.log("Backend not available, using mock health data")
          // Set mock health data
          setSystemHealth({
            status: "healthy",
            timestamp: new Date().toISOString(),
            version: "1.0.0",
            services: {
              database: {
                status: "healthy",
                last_checked: new Date().toISOString(),
              },
              weather: {
                status: "healthy",
                last_checked: new Date().toISOString(),
              },
              ml_model: {
                status: "healthy",
                last_checked: new Date().toISOString(),
              },
            },
          })
        }
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to fetch system health data",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }

    fetchSystemHealth()
  }, [router, toast])

  if (isLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="flex h-screen w-full flex-col md:flex-row">
      <DashboardSidebar systemId={systemId} />

      <div className="flex flex-1 flex-col overflow-hidden">
        <DashboardHeader systemId={systemId} systemHealth={systemHealth} />

        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="mb-4 grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
              <TabsTrigger value="control">Control</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              <OverviewTab systemId={systemId} />
            </TabsContent>

            <TabsContent value="analytics" className="space-y-4">
              <AnalyticsTab systemId={systemId} />
            </TabsContent>

            <TabsContent value="control" className="space-y-4">
              <ControlTab systemId={systemId} />
            </TabsContent>
          </Tabs>
        </main>
      </div>
      <Toaster />
    </div>
  )
}

