"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useToast } from "@/components/ui/use-toast"
import { Loader2, Thermometer, Droplets, Zap, Gauge } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"
import ACTable from "@/components/ac-table"

interface OverviewTabProps {
  systemId: string
}

export default function OverviewTab({ systemId }: OverviewTabProps) {
  const { toast } = useToast()
  const [isLoading, setIsLoading] = useState(true)
  const [currentData, setCurrentData] = useState<any>(null)
  const [historyData, setHistoryData] = useState<any[]>([])
  const [metricsData, setMetricsData] = useState<any>(null)

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      const token = localStorage.getItem("token")

      try {
        // Fetch current temperature
        const tempResponse = await fetch(
          `http://localhost:8000/api/temperature/current?device_id=${systemId}&zone_id=main`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          },
        )

        if (!tempResponse.ok) {
          throw new Error("Failed to fetch current temperature")
        }

        const tempData = await tempResponse.json()
        setCurrentData(tempData)

        // Fetch temperature history
        const historyResponse = await fetch(
          `http://localhost:8000/api/temperature/history?device_id=${systemId}&zone_id=main`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          },
        )

        if (!historyResponse.ok) {
          throw new Error("Failed to fetch temperature history")
        }

        const historyData = await historyResponse.json()
        setHistoryData(historyData.history || [])

        // Fetch system metrics
        const metricsResponse = await fetch(`http://localhost:8000/api/status/metrics?system_id=${systemId}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        })

        if (!metricsResponse.ok) {
          throw new Error("Failed to fetch system metrics")
        }

        const metricsData = await metricsResponse.json()
        setMetricsData(metricsData)
      } catch (error) {
        console.log("Error fetching data, using mock data", error)

        // Set mock current data if API call fails
        if (!currentData) {
          setCurrentData({
            temperature: 23.5,
            humidity: 50.0,
            timestamp: new Date().toISOString(),
            device_id: systemId,
            zone_id: "main",
          })
        }

        // Set mock history data if API call fails
        if (historyData.length === 0) {
          const mockHistory = Array.from({ length: 24 }, (_, i) => ({
            temperature: 22 + Math.random() * 3,
            humidity: 45 + Math.random() * 10,
            timestamp: new Date(Date.now() - (23 - i) * 3600000).toISOString(),
            device_id: systemId,
            zone_id: "main",
          }))
          setHistoryData(mockHistory)
        }

        // Set mock metrics data if API call fails
        if (!metricsData) {
          setMetricsData({
            data: {
              summary: {
                avg_power: 1150,
                total_energy: 426.9,
                efficiency: 0.85,
              },
            },
          })
        }

        toast({
          title: "Using Mock Data",
          description: "Backend not available, displaying mock data",
          variant: "default",
        })
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()

    // Set up polling every 30 seconds
    const interval = setInterval(fetchData, 30000)

    return () => clearInterval(interval)
  }, [systemId, toast, currentData, historyData.length, metricsData])

  // Mock data for the charts
  const mockPowerData = [
    { time: "00:00", power: 980, voltage: 220, current: 4.5, frequency: 50.1 },
    { time: "01:00", power: 1020, voltage: 221, current: 4.6, frequency: 50.0 },
    { time: "02:00", power: 990, voltage: 219, current: 4.5, frequency: 50.2 },
    { time: "03:00", power: 950, voltage: 220, current: 4.3, frequency: 50.1 },
    { time: "04:00", power: 930, voltage: 218, current: 4.3, frequency: 50.0 },
    { time: "05:00", power: 920, voltage: 219, current: 4.2, frequency: 50.1 },
    { time: "06:00", power: 950, voltage: 220, current: 4.3, frequency: 50.0 },
    { time: "07:00", power: 1050, voltage: 221, current: 4.8, frequency: 49.9 },
    { time: "08:00", power: 1150, voltage: 222, current: 5.2, frequency: 49.8 },
    { time: "09:00", power: 1200, voltage: 223, current: 5.4, frequency: 49.9 },
    { time: "10:00", power: 1250, voltage: 224, current: 5.6, frequency: 50.0 },
    { time: "11:00", power: 1300, voltage: 225, current: 5.8, frequency: 50.1 },
  ]

  const mockEfficiencyData = [
    { time: "00:00", efficiency: 0.82 },
    { time: "01:00", efficiency: 0.83 },
    { time: "02:00", efficiency: 0.84 },
    { time: "03:00", efficiency: 0.85 },
    { time: "04:00", efficiency: 0.86 },
    { time: "05:00", efficiency: 0.87 },
    { time: "06:00", efficiency: 0.86 },
    { time: "07:00", efficiency: 0.85 },
    { time: "08:00", efficiency: 0.84 },
    { time: "09:00", efficiency: 0.83 },
    { time: "10:00", efficiency: 0.82 },
    { time: "11:00", efficiency: 0.81 },
  ]

  if (isLoading) {
    return (
      <div className="flex h-64 w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Temperature</CardTitle>
            <Thermometer className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentData?.temperature || "--"}°C</div>
            <p className="text-xs text-muted-foreground">
              Last updated: {new Date(currentData?.timestamp || Date.now()).toLocaleTimeString()}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Humidity</CardTitle>
            <Droplets className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentData?.humidity || "--"}%</div>
            <p className="text-xs text-muted-foreground">Optimal range: 40-60%</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Power Consumption</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metricsData?.data?.summary?.avg_power || "--"} W</div>
            <p className="text-xs text-muted-foreground">
              Total energy: {metricsData?.data?.summary?.total_energy || "--"} kWh
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Efficiency</CardTitle>
            <Gauge className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(metricsData?.data?.summary?.efficiency || 0.85) * 100}%</div>
            <p className="text-xs text-muted-foreground">
              {(metricsData?.data?.summary?.efficiency || 0.85) > 0.8 ? "Good" : "Needs improvement"}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Temperature History</CardTitle>
            <CardDescription>Last 24 hours temperature readings</CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={historyData.length > 0 ? historyData : mockPowerData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(timestamp) => {
                    const date = new Date(timestamp)
                    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, "0")}`
                  }}
                />
                <YAxis />
                <Tooltip
                  labelFormatter={(timestamp) => {
                    return new Date(timestamp).toLocaleString()
                  }}
                />
                <Line type="monotone" dataKey="temperature" stroke="#8884d8" activeDot={{ r: 8 }} />
                <Line type="monotone" dataKey="humidity" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Power Metrics</CardTitle>
            <CardDescription>Power, voltage, current, and frequency</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="power">
              <TabsList className="mb-4 grid w-full grid-cols-4">
                <TabsTrigger value="power">Power</TabsTrigger>
                <TabsTrigger value="voltage">Voltage</TabsTrigger>
                <TabsTrigger value="current">Current</TabsTrigger>
                <TabsTrigger value="frequency">Frequency</TabsTrigger>
              </TabsList>

              <TabsContent value="power" className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mockPowerData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="power" stroke="#8884d8" fill="#8884d8" />
                  </AreaChart>
                </ResponsiveContainer>
              </TabsContent>

              <TabsContent value="voltage" className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mockPowerData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis domain={[210, 230]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="voltage" stroke="#82ca9d" />
                  </LineChart>
                </ResponsiveContainer>
              </TabsContent>

              <TabsContent value="current" className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mockPowerData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis domain={[4, 6]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="current" stroke="#ffc658" />
                  </LineChart>
                </ResponsiveContainer>
              </TabsContent>

              <TabsContent value="frequency" className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={mockPowerData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis domain={[49.5, 50.5]} />
                    <Tooltip />
                    <Line type="monotone" dataKey="frequency" stroke="#ff7300" />
                  </LineChart>
                </ResponsiveContainer>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Efficiency Trend</CardTitle>
          <CardDescription>System efficiency over time</CardDescription>
        </CardHeader>
        <CardContent className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={mockEfficiencyData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[0.7, 0.9]} />
              <Tooltip />
              <Area type="monotone" dataKey="efficiency" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Connected AC Systems</CardTitle>
          <CardDescription>Overview of all connected AC units</CardDescription>
        </CardHeader>
        <CardContent>
          <ACTable systemId={systemId} />
        </CardContent>
      </Card>
    </div>
  )
}

