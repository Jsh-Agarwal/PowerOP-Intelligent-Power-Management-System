import { NavigationContainer } from "@react-navigation/native"; 
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { ThemeProvider } from "./components/ThemeContext";
import LoginScreen from "./screens/LoginScreen";
import UserNavigator from "./navigation/UserNavigator";
import AdminNavigator from "./navigation/AdminNavigator";

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <ThemeProvider>
      <NavigationContainer>
        <Stack.Navigator initialRouteName="Login" screenOptions={{ headerShown: false }}>
          <Stack.Screen name="Login" component={LoginScreen} />
          <Stack.Screen name="UserApp" component={UserNavigator} />
          <Stack.Screen name="AdminApp" component={AdminNavigator} />
        </Stack.Navigator>
      </NavigationContainer>
    </ThemeProvider>
  );
}
