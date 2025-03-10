"use client"

import { useRouter } from "next/navigation"
import { BarChart3, Home, Settings, Thermometer, Wind, Zap, Activity, AreaChart, Gauge } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"

interface DashboardSidebarProps {
  systemId: string
}

export default function DashboardSidebar({ systemId }: DashboardSidebarProps) {
  const router = useRouter()

  const handleSystemChange = (newSystemId: string) => {
    localStorage.setItem("selectedAC", newSystemId)
    router.push(`/dashboard/${newSystemId}`)
  }

  return (
    <div className="hidden border-r bg-muted/40 md:block md:w-64">
      <div className="flex h-16 items-center border-b px-4">
        <Wind className="mr-2 h-6 w-6 text-primary" />
        <h2 className="text-lg font-semibold">PowerOP ML</h2>
      </div>
      <ScrollArea className="h-[calc(100vh-4rem)]">
        <div className="px-3 py-2">
          <h3 className="mb-2 px-4 text-xs font-semibold uppercase text-muted-foreground">Navigation</h3>
          <div className="space-y-1">
            <Button
              variant="ghost"
              className="w-full justify-start"
              onClick={() => router.push(`/dashboard/${systemId}`)}
            >
              <Home className="mr-2 h-4 w-4" />
              Dashboard
            </Button>
            <Button
              variant="ghost"
              className="w-full justify-start"
              onClick={() => router.push(`/dashboard/${systemId}?tab=analytics`)}
            >
              <BarChart3 className="mr-2 h-4 w-4" />
              Analytics
            </Button>
            <Button
              variant="ghost"
              className="w-full justify-start"
              onClick={() => router.push(`/dashboard/${systemId}?tab=control`)}
            >
              <Settings className="mr-2 h-4 w-4" />
              Control
            </Button>
          </div>
        </div>

        <div className="px-3 py-2">
          <h3 className="mb-2 px-4 text-xs font-semibold uppercase text-muted-foreground">AC Systems</h3>
          <div className="space-y-1">
            <Button
              variant={systemId === "ac1" ? "secondary" : "ghost"}
              className="w-full justify-start"
              onClick={() => handleSystemChange("ac1")}
            >
              <Thermometer className="mr-2 h-4 w-4" />
              AC System 1
            </Button>
            <Button
              variant={systemId === "ac2" ? "secondary" : "ghost"}
              className="w-full justify-start"
              onClick={() => handleSystemChange("ac2")}
            >
              <Thermometer className="mr-2 h-4 w-4" />
              AC System 2
            </Button>
            <Button
              variant={systemId === "ac3" ? "secondary" : "ghost"}
              className="w-full justify-start"
              onClick={() => handleSystemChange("ac3")}
            >
              <Thermometer className="mr-2 h-4 w-4" />
              AC System 3
            </Button>
          </div>
        </div>

        <div className="px-3 py-2">
          <h3 className="mb-2 px-4 text-xs font-semibold uppercase text-muted-foreground">Monitoring</h3>
          <div className="space-y-1">
            <Button variant="ghost" className="w-full justify-start">
              <Zap className="mr-2 h-4 w-4" />
              Power Metrics
            </Button>
            <Button variant="ghost" className="w-full justify-start">
              <Activity className="mr-2 h-4 w-4" />
              Performance
            </Button>
            <Button variant="ghost" className="w-full justify-start">
              <AreaChart className="mr-2 h-4 w-4" />
              Energy Usage
            </Button>
            <Button variant="ghost" className="w-full justify-start">
              <Gauge className="mr-2 h-4 w-4" />
              Efficiency
            </Button>
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}

