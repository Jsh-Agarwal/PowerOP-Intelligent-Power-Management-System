"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { ModeToggle } from "@/components/mode-toggle"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Bell, LogOut, Menu, Wind, CheckCircle2, AlertCircle, XCircle } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import DashboardSidebar from "@/components/dashboard-sidebar"

interface DashboardHeaderProps {
  systemId: string
  systemHealth: any
}

export default function DashboardHeader({ systemId, systemHealth }: DashboardHeaderProps) {
  const router = useRouter()
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => {
      clearInterval(timer)
    }
  }, [])

  const handleLogout = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("selectedAC")
    router.push("/")
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />
      case "warning":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />
      default:
        return <XCircle className="h-4 w-4 text-red-500" />
    }
  }

  return (
    <header className="sticky top-0 z-10 flex h-16 items-center justify-between border-b bg-background px-4 md:px-6">
      <div className="flex items-center gap-2 md:hidden">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="outline" size="icon">
              <Menu className="h-5 w-5" />
              <span className="sr-only">Toggle Menu</span>
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="p-0">
            <DashboardSidebar systemId={systemId} />
          </SheetContent>
        </Sheet>
      </div>

      <div className="flex items-center gap-2">
        <Wind className="h-6 w-6 text-primary" />
        <h1 className="text-lg font-semibold md:text-xl">AC System {systemId.replace("ac", "")}</h1>
        {systemHealth && (
          <Badge variant={systemHealth.status === "healthy" ? "outline" : "destructive"} className="ml-2 gap-1">
            {getStatusIcon(systemHealth.status)}
            {systemHealth.status}
          </Badge>
        )}
      </div>

      <div className="hidden md:block">
        <p className="text-sm text-muted-foreground">{currentTime.toLocaleString()}</p>
      </div>

      <div className="flex items-center gap-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon">
              <Bell className="h-5 w-5" />
              <span className="sr-only">Notifications</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Notifications</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem>System maintenance scheduled</DropdownMenuItem>
            <DropdownMenuItem>Temperature anomaly detected</DropdownMenuItem>
            <DropdownMenuItem>Energy usage report available</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <ModeToggle />

        <Button variant="ghost" size="icon" onClick={handleLogout}>
          <LogOut className="h-5 w-5" />
          <span className="sr-only">Log out</span>
        </Button>
      </div>
    </header>
  )
}

