using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using Windows.Storage.Pickers;

// https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x804 上介绍了“空白页”项模板

namespace ObjectAndPoseDetection.UWP
{
    /// <summary>
    /// 可用于自身或导航至 Frame 内部的空白页。
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            this.InitializeComponent();
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FolderPicker();
            var chosenFolder = await picker.PickSingleFolderAsync();

            if(chosenFolder != null)
            {
                //DetectObjectPose(chosenFolder.Path);
            }
        }

        private void DetectObjectPose(string path)
        {
            //ObjectDetector objectDetector = new ObjectDetector("../Assets/OnnxModel/MultiObjectDetectionModel.onnx");
            //var boxes = objectDetector.DetectFromFolder(path);
        }
    }
}
