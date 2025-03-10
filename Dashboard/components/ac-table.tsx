"use client"

import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Eye, ThermometerSun, Zap, Gauge, Activity, Power, Droplets } from "lucide-react"
import ACControlPanel from "@/components/ac-control-panel"

interface ACTableProps {
  systemId: string
}

export default function ACTable({ systemId }: ACTableProps) {
  const [selectedAC, setSelectedAC] = useState<string | null>(null)

  // Mock data for the AC systems
  const acSystems = [
    {
      id: "ac1",
      name: "AC System 1",
      zone: "Main Floor",
      status: "Running",
      temperature: 22.5,
      targetTemp: 22.0,
      humidity: 45,
      power: 1200,
      voltage: 220,
      current: 5.5,
      frequency: 50.1,
      efficiency: 0.85,
    },
    {
      id: "ac2",
      name: "AC System 2",
      zone: "Second Floor",
      status: "Running",
      temperature: 23.2,
      targetTemp: 23.0,
      humidity: 48,
      power: 1150,
      voltage: 221,
      current: 5.2,
      frequency: 50.0,
      efficiency: 0.83,
    },
    {
      id: "ac3",
      name: "AC System 3",
      zone: "Basement",
      status: "Idle",
      temperature: 24.0,
      targetTemp: 24.0,
      humidity: 52,
      power: 50,
      voltage: 220,
      current: 0.2,
      frequency: 50.0,
      efficiency: 0.8,
    },
  ]

  const handleViewMore = (acId: string) => {
    setSelectedAC(acId)
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "Running":
        return <Badge className="bg-green-500">Running</Badge>
      case "Idle":
        return <Badge variant="outline">Idle</Badge>
      case "Error":
        return <Badge variant="destructive">Error</Badge>
      default:
        return <Badge variant="secondary">{status}</Badge>
    }
  }

  const selectedACData = acSystems.find((ac) => ac.id === selectedAC)

  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Zone</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="hidden md:table-cell">Temperature</TableHead>
            <TableHead className="hidden md:table-cell">Power</TableHead>
            <TableHead className="hidden lg:table-cell">Efficiency</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {acSystems.map((ac) => (
            <TableRow key={ac.id}>
              <TableCell className="font-medium">{ac.name}</TableCell>
              <TableCell>{ac.zone}</TableCell>
              <TableCell>{getStatusBadge(ac.status)}</TableCell>
              <TableCell className="hidden md:table-cell">
                {ac.temperature}°C / {ac.targetTemp}°C
              </TableCell>
              <TableCell className="hidden md:table-cell">{ac.power} W</TableCell>
              <TableCell className="hidden lg:table-cell">{(ac.efficiency * 100).toFixed(1)}%</TableCell>
              <TableCell className="text-right">
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" className="h-8 gap-1" onClick={() => handleViewMore(ac.id)}>
                      <Eye className="h-3.5 w-3.5" />
                      <span className="hidden sm:inline">View More</span>
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-[600px]">
                    <DialogHeader>
                      <DialogTitle>{selectedACData?.name}</DialogTitle>
                      <DialogDescription>
                        Zone: {selectedACData?.zone} | Status: {selectedACData?.status}
                      </DialogDescription>
                    </DialogHeader>

                    <Tabs defaultValue="control" className="mt-4">
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="control">Control</TabsTrigger>
                        <TabsTrigger value="metrics">Metrics</TabsTrigger>
                      </TabsList>

                      <TabsContent value="control" className="space-y-4 py-4">
                        <ACControlPanel acId={selectedAC || ""} />
                      </TabsContent>

                      <TabsContent value="metrics" className="py-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <ThermometerSun className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm font-medium">Temperature</span>
                            </div>
                            <div className="text-2xl font-bold">{selectedACData?.temperature}°C</div>
                            <div className="text-xs text-muted-foreground">Target: {selectedACData?.targetTemp}°C</div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Droplets className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm font-medium">Humidity</span>
                            </div>
                            <div className="text-2xl font-bold">{selectedACData?.humidity}%</div>
                            <div className="text-xs text-muted-foreground">Optimal: 40-60%</div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Zap className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm font-medium">Power</span>
                            </div>
                            <div className="text-2xl font-bold">{selectedACData?.power} W</div>
                            <div className="text-xs text-muted-foreground">Voltage: {selectedACData?.voltage}V</div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Activity className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm font-medium">Current</span>
                            </div>
                            <div className="text-2xl font-bold">{selectedACData?.current} A</div>
                            <div className="text-xs text-muted-foreground">
                              Frequency: {selectedACData?.frequency} Hz
                            </div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Gauge className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm font-medium">Efficiency</span>
                            </div>
                            <div className="text-2xl font-bold">{(selectedACData?.efficiency || 0) * 100}%</div>
                            <div className="text-xs text-muted-foreground">
                              {(selectedACData?.efficiency || 0) > 0.8 ? "Good" : "Needs improvement"}
                            </div>
                          </div>

                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <Power className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm font-medium">Status</span>
                            </div>
                            <div className="text-2xl font-bold">{selectedACData?.status}</div>
                            <div className="text-xs text-muted-foreground">
                              Since: {new Date().toLocaleTimeString()}
                            </div>
                          </div>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </DialogContent>
                </Dialog>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </>
  )
}

