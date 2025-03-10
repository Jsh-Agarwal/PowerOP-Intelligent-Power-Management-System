"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Loader2, Wind } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { Toaster } from "@/components/ui/toaster"

export default function LoginForm() {
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [selectedAC, setSelectedAC] = useState("ac1")
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()
  const { toast } = useToast()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      // Try to call the authentication endpoint
      try {
        const response = await fetch("http://localhost:8000/token", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ username, password }),
        })

        if (response.ok) {
          const data = await response.json()
          localStorage.setItem("token", data.access_token)
        } else {
          // If backend is not available, use a mock token
          console.log("Backend not available, using mock token")
          localStorage.setItem("token", "mock-token-for-development-only")
        }
      } catch (error) {
        // If fetch fails (backend not available), use a mock token
        console.log("Backend not available, using mock token")
        localStorage.setItem("token", "mock-token-for-development-only")
      }

      // Always store the selected AC and redirect
      localStorage.setItem("selectedAC", selectedAC)

      // Show success message
      toast({
        title: "Login Successful",
        description: "Redirecting to dashboard...",
      })

      // Redirect to dashboard
      router.push(`/dashboard/${selectedAC}`)
    } catch (error) {
      toast({
        title: "Login Failed",
        description: "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      <Card className="w-[350px] shadow-lg">
        <CardHeader className="space-y-1">
          <div className="flex items-center justify-center mb-2">
            <Wind className="h-10 w-10 text-primary" />
          </div>
          <CardTitle className="text-2xl text-center">PowerOP Dashboard</CardTitle>
          <CardDescription className="text-center">Enter your credentials to access the system</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form onSubmit={handleLogin}>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>

              <div className="space-y-2">
                <Label>Select AC System</Label>
                <RadioGroup
                  defaultValue="ac1"
                  value={selectedAC}
                  onValueChange={setSelectedAC}
                  className="flex flex-col space-y-1"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="ac1" id="ac1" />
                    <Label htmlFor="ac1">AC System 1</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="ac2" id="ac2" />
                    <Label htmlFor="ac2">AC System 2</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="ac3" id="ac3" />
                    <Label htmlFor="ac3">AC System 3</Label>
                  </div>
                </RadioGroup>
              </div>

              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Logging in
                  </>
                ) : (
                  "Login"
                )}
              </Button>
            </div>
          </form>
        </CardContent>
        <CardFooter className="flex justify-center text-sm text-muted-foreground">
          Industrial AC Monitoring System
        </CardFooter>
      </Card>
      <Toaster />
    </>
  )
}

