"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/use-toast"
import { Calendar, Loader2, Download, AlertTriangle, TrendingUp, DollarSign, Lightbulb } from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface AnalyticsTabProps {
  systemId: string
}

export default function AnalyticsTab({ systemId }: AnalyticsTabProps) {
  const { toast } = useToast()
  const [isLoading, setIsLoading] = useState(true)
  const [costData, setCostData] = useState<any>(null)
  const [timeRange, setTimeRange] = useState("week")

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      const token = localStorage.getItem("token")

      try {
        // Fetch cost analysis data
        const costResponse = await fetch(`http://localhost:8000/api/analysis/cost/${systemId}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        })

        if (!costResponse.ok) {
          throw new Error("Failed to fetch cost analysis")
        }

        const costData = await costResponse.json()
        setCostData(costData)
      } catch (error) {
        console.log("Error fetching analytics data, using mock data", error)

        // Set mock cost data if API call fails
        setCostData({
          analysis: {
            total_energy_kwh: 426919.1,
            total_cost: 64037.87,
            average_daily_cost: 9094.13,
            peak_usage_kwh: 426919.1,
            peak_usage_cost: 64037.87,
            efficiency_score: 0.85,
          },
          recommendations: [
            {
              type: "peak_usage",
              message: "Consider shifting load to off-peak hours",
              potential_savings: "10-15%",
            },
            {
              type: "maintenance",
              message: "Schedule regular maintenance",
              potential_savings: "5-8%",
            },
          ],
        })

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
  }, [systemId, toast])

  // Mock data for the charts
  const mockEnergyData = [
    { day: "Mon", energy: 120 },
    { day: "Tue", energy: 132 },
    { day: "Wed", energy: 101 },
    { day: "Thu", energy: 134 },
    { day: "Fri", energy: 90 },
    { day: "Sat", energy: 70 },
    { day: "Sun", energy: 85 },
  ]

  const mockCostData = [
    { day: "Mon", cost: 18 },
    { day: "Tue", cost: 19.8 },
    { day: "Wed", cost: 15.15 },
    { day: "Thu", cost: 20.1 },
    { day: "Fri", cost: 13.5 },
    { day: "Sat", cost: 10.5 },
    { day: "Sun", cost: 12.75 },
  ]

  const mockAnomalyData = [
    { time: "00:00", temperature: 22.5, isAnomaly: false },
    { time: "01:00", temperature: 22.7, isAnomaly: false },
    { time: "02:00", temperature: 22.8, isAnomaly: false },
    { time: "03:00", temperature: 22.6, isAnomaly: false },
    { time: "04:00", temperature: 22.5, isAnomaly: false },
    { time: "05:00", temperature: 22.4, isAnomaly: false },
    { time: "06:00", temperature: 22.3, isAnomaly: false },
    { time: "07:00", temperature: 22.5, isAnomaly: false },
    { time: "08:00", temperature: 22.8, isAnomaly: false },
    { time: "09:00", temperature: 23.2, isAnomaly: false },
    { time: "10:00", temperature: 25.5, isAnomaly: true },
    { time: "11:00", temperature: 23.5, isAnomaly: false },
  ]

  const mockEfficiencyData = [
    { name: "Optimal", value: 65 },
    { name: "Suboptimal", value: 25 },
    { name: "Inefficient", value: 10 },
  ]

  const COLORS = ["#0088FE", "#FFBB28", "#FF8042"]

  if (isLoading) {
    return (
      <div className="flex h-64 w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Analytics Dashboard</h2>
        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="day">Last 24 Hours</SelectItem>
              <SelectItem value="week">Last Week</SelectItem>
              <SelectItem value="month">Last Month</SelectItem>
              <SelectItem value="year">Last Year</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon">
            <Download className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon">
            <Calendar className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Energy</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{costData?.analysis?.total_energy_kwh?.toFixed(1) || "--"} kWh</div>
            <p className="text-xs text-muted-foreground">+2.5% from last period</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${costData?.analysis?.total_cost?.toFixed(2) || "--"}</div>
            <p className="text-xs text-muted-foreground">
              Average daily: ${costData?.analysis?.average_daily_cost?.toFixed(2) || "--"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Efficiency Score</CardTitle>
            <Lightbulb className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{((costData?.analysis?.efficiency_score || 0) * 100).toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              {(costData?.analysis?.efficiency_score || 0) > 0.8 ? "Good" : "Needs improvement"}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Anomalies</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1</div>
            <p className="text-xs text-muted-foreground">Last detected: Today at 10:00</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Energy Consumption</CardTitle>
            <CardDescription>Daily energy usage in kWh</CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockEnergyData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="energy" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Cost Analysis</CardTitle>
            <CardDescription>Daily cost in dollars</CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockCostData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="cost" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Anomaly Detection</CardTitle>
            <CardDescription>Temperature anomalies over time</CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockAnomalyData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[20, 30]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="temperature"
                  stroke="#8884d8"
                  dot={(props) => {
                    const { cx, cy, payload } = props
                    if (payload.isAnomaly) {
                      return <circle cx={cx} cy={cy} r={6} fill="red" stroke="none" />
                    }
                    return <circle cx={cx} cy={cy} r={4} fill="#8884d8" stroke="none" />
                  }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Efficiency Distribution</CardTitle>
            <CardDescription>Operational efficiency breakdown</CardDescription>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={mockEfficiencyData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {mockEfficiencyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Optimization Recommendations</CardTitle>
          <CardDescription>AI-powered suggestions to improve efficiency</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {costData?.recommendations?.map((rec: any, index: number) => (
              <div key={index} className="rounded-lg border p-4">
                <div className="flex items-start gap-4">
                  <div className="rounded-full bg-primary/10 p-2">
                    <Lightbulb className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-medium">{rec.type}</h3>
                    <p className="text-sm text-muted-foreground">{rec.message}</p>
                    <p className="mt-1 text-sm font-medium">Potential savings: {rec.potential_savings}</p>
                  </div>
                </div>
              </div>
            ))}

            {(!costData?.recommendations || costData.recommendations.length === 0) && (
              <div className="rounded-lg border p-4">
                <div className="flex items-start gap-4">
                  <div className="rounded-full bg-primary/10 p-2">
                    <Lightbulb className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-medium">Peak Usage Optimization</h3>
                    <p className="text-sm text-muted-foreground">
                      Consider shifting load to off-peak hours to reduce energy costs.
                    </p>
                    <p className="mt-1 text-sm font-medium">Potential savings: 10-15%</p>
                  </div>
                </div>
              </div>
            )}

            <div className="rounded-lg border p-4">
              <div className="flex items-start gap-4">
                <div className="rounded-full bg-primary/10 p-2">
                  <Lightbulb className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-medium">Temperature Setpoint Adjustment</h3>
                  <p className="text-sm text-muted-foreground">
                    Increasing cooling setpoint by 1°C can reduce energy consumption by approximately 3-5%.
                  </p>
                  <p className="mt-1 text-sm font-medium">Potential savings: 3-5%</p>
                </div>
              </div>
            </div>

            <div className="rounded-lg border p-4">
              <div className="flex items-start gap-4">
                <div className="rounded-full bg-primary/10 p-2">
                  <Lightbulb className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-medium">Maintenance Schedule</h3>
                  <p className="text-sm text-muted-foreground">
                    Regular maintenance can improve efficiency. Schedule a filter cleaning within the next 2 weeks.
                  </p>
                  <p className="mt-1 text-sm font-medium">Potential savings: 5-10%</p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

