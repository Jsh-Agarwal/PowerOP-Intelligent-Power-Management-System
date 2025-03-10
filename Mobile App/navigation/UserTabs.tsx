import { createBottomTabNavigator } from "@react-navigation/bottom-tabs"
import { Ionicons } from "@expo/vector-icons"
import HomeScreen from "../screens/HomeScreen"
import StatsScreen from "../screens/StatsScreen"
import CalendarScreen from "../screens/CalendarScreen"
import ProfileScreen from "../screens/ProfileScreen"
import { useTheme } from "../components/ThemeContext"

const Tab = createBottomTabNavigator()

const UserTabs = () => {
  const { theme } = useTheme()

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap

          if (route.name === "Home") {
            iconName = focused ? "home" : "home-outline"
          } else if (route.name === "Stats") {
            iconName = focused ? "stats-chart" : "stats-chart-outline"
          } else if (route.name === "Calendar") {
            iconName = focused ? "calendar" : "calendar-outline"
          } else if (route.name === "Profile") {
            iconName = focused ? "person" : "person-outline"
          } else {
            iconName = "home" // default icon
          }

          return <Ionicons name={iconName} size={size} color={color} />
        },
        tabBarActiveTintColor: theme === "dark" ? "#BB86FC" : "#6200EE",
        tabBarInactiveTintColor: "gray",
        tabBarStyle: {
          backgroundColor: theme === "dark" ? "#121212" : "#FFFFFF",
        },
        headerStyle: {
          backgroundColor: theme === "dark" ? "#121212" : "#FFFFFF",
        },
        headerTintColor: theme === "dark" ? "#FFFFFF" : "#000000",
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Stats" component={StatsScreen} />
      <Tab.Screen name="Calendar" component={CalendarScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  )
}

export default UserTabs

