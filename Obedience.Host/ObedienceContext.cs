using System;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.Net;
using System.Windows.Forms;
using Sonda.Sensors;

namespace Obedience.Host
{
    internal class ObedienceContext : ApplicationContext
    {
        private Container _components;
        private NotifyIcon _notifyIcon;

        private AcdController scanner;

        internal ObedienceContext()
        {
            //Instantiate the component Module to hold everything
            _components = new System.ComponentModel.Container();


            //Instantiate the NotifyIcon attaching it to the components container and 
            //provide it an icon, note, you can imbed this resource 
            _notifyIcon = new NotifyIcon(_components);
            _notifyIcon.Icon = Resources.AppIcon;
            _notifyIcon.Text = "Obedience";
            _notifyIcon.Visible = true;

            //Instantiate the context menu and items
            var contextMenu = new ContextMenuStrip();
            var displayForm = new ToolStripMenuItem();
            var exitApplication = new ToolStripMenuItem();

            //Attach the menu to the notify icon
            _notifyIcon.ContextMenuStrip = contextMenu;

            //Setup the items and add them to the menu strip, adding handlers to be created later
            displayForm.Text = "Do something";
            displayForm.Click += mDisplayForm_Click;
            contextMenu.Items.Add(displayForm);

            exitApplication.Text = "Exit";
            exitApplication.Click += mExitApplication_Click;
            contextMenu.Items.Add(exitApplication);

            scanner = new AcdController(new IPEndPoint(new IPAddress(new byte[] {10, 1, 0, 220}), 5003));
            scanner.Reboot();
            scanner.ConnectedChanged += new ConnectedChangedEventHandler(scanner_ConnectedChanged);
        }

        void scanner_ConnectedChanged(bool connectedStatus)
        {
            if (connectedStatus)
                scanner.Sensor.Captured += new Sonda.NET.CapturedEventHandler<Sonda.NET.SondaImage>(Sensor_Captured);
            else scanner.Sensor.Captured -= Sensor_Captured;
        }

        void Sensor_Captured(object sender, Sonda.NET.SondaImage captureResult)
        {
            var bmp = captureResult.ToBitmap();
            var endbmp = new Bitmap(256, 364);
            for (int x = 0; x < 256; x++)
            {
                for (int y = 0; y < 364; y++)
                {
                    var color = bmp.GetPixel(x + 300 - 256, y + 384 - 364);
                    endbmp.SetPixel(x, y, color);
                }   
            }
            endbmp.Save("C:\\temp\\acd.png", ImageFormat.Png);
        }

        private void mExitApplication_Click(object sender, EventArgs e)
        {
            _notifyIcon.Visible = false;
            scanner.Sensor.Captured -= Sensor_Captured;
            scanner.Dispose();
            ExitThreadCore();
        }

        private void mDisplayForm_Click(object sender, EventArgs e)
        {
            ExitThreadCore();
        }
    }
}
