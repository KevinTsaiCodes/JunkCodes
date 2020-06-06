//author: 程式碼醫生工作室
//公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
//來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
//配套程式碼技術支援：bbs.aianaconda.com      (有問必答)

#import "13-3 CameraExampleAppDelegate.h"
@implementation _3_3_CameraExampleAppDelegate

@synthesize window = _window;

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
  [self.window makeKeyAndVisible];
  return YES;
}

- (void)applicationWillResignActive:(UIApplication *)application {
  [[UIApplication sharedApplication] setIdleTimerDisabled:NO];
}

- (void)applicationDidEnterBackground:(UIApplication *)application {
}

- (void)applicationWillEnterForeground:(UIApplication *)application {
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
  [[UIApplication sharedApplication] setIdleTimerDisabled:YES];
}

- (void)applicationWillTerminate:(UIApplication *)application {
}

@end
