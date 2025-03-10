"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useToast } from "@/components/ui/use-toast"
import { Loader2 } from "lucide-react"
import ACControlPanel from "@/components/ac-control-panel"

interface ControlTabProps {
  systemId: string
}

export default function ControlTab({ systemId }: ControlTabProps) {
  const { toast } = useToast()
  const [isLoading, setIsLoading] = useState(false)

  if (isLoading) {
    return (
      <div className="flex h-64 w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>AC System Controls</CardTitle>
          <CardDescription>Manage and monitor your AC system settings</CardDescription>
        </CardHeader>
        <CardContent>
          <ACControlPanel acId={systemId} />
        </CardContent>
      </Card>
    </div>
  )
}

